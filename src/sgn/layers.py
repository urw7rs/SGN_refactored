from typing import List

from einops import parse_shape, rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import Sequential


class DynamicsRepresentation(nn.Module):
    def __init__(self, num_features, num_joints):
        super().__init__()

        self.batch_norm1 = nn.BatchNorm1d(num_features * num_joints)
        self.batch_norm2 = nn.BatchNorm1d(num_features * num_joints)

        self.embed_pos = Embed([num_features, 64, 64])
        self.embed_vel = Embed([num_features, 64, 64])

    def forward(self, joint):
        zero = torch.zeros_like(joint[:, 0:1])
        padded_position = torch.cat([zero, joint], dim=1)
        velocity = padded_position[:, 1:] - padded_position[:, :-1]

        shape = parse_shape(joint, "n t v _")

        joint = rearrange(joint, "n t v c -> (n t) (v c)")
        joint = self.batch_norm1(joint)
        joint = rearrange(joint, "(n t) (v c) -> n t v c", **shape)
        pos_embedding = self.embed_pos(joint)

        velocity = rearrange(velocity, "n t v c -> (n t) (v c)")
        velocity = self.batch_norm2(velocity)
        velocity = rearrange(velocity, "(n t) (v c) -> n t v c", **shape)
        vel_embedding = self.embed_vel(velocity)

        fused_embedding = pos_embedding + vel_embedding

        return fused_embedding


class JointLevelModule(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        joint_indices = torch.arange(0, num_joints)
        one_hot_joint = F.one_hot(joint_indices, num_classes=num_joints).float()
        one_hot_joint = one_hot_joint.view(num_joints, num_joints)
        self.register_buffer("j", one_hot_joint)

        self.embed_joint = Embed([num_joints, 64, 64])

        self.compute_adjacency_matrix = AdjacencyMatrix(128, 256)

        self.gcn1 = self.build_gcn(128, 128)
        self.gcn2 = self.build_gcn(128, 256)
        self.gcn3 = self.build_gcn(256, 256)

    def forward(self, z):
        N, V, C = z.size()

        j = repeat(self.j, "v1 v2 -> n v1 v2", n=N)
        z = torch.cat([z, self.embed_joint(j)], dim=-1)

        # G: N, V, V
        G = self.compute_adjacency_matrix(z)

        # x: N, V, C, adj: N, V, V
        x = self.gcn1(z, G)
        x = self.gcn2(x, G)
        x = self.gcn3(x, G)

        return x

    def build_gcn(self, in_channels, out_channels):
        return Sequential(
            "x, adj",
            [
                (
                    ResidualGCN(in_channels, out_channels),
                    "x, adj -> x",
                ),
                Rearrange("n v c -> n c v"),
                nn.BatchNorm1d(out_channels),
                Rearrange("n c v -> n v c"),
                nn.ReLU(inplace=True),
            ],
        )


class FrameLevelModule(nn.Module):
    def __init__(self, length):
        super().__init__()

        frame_indices = torch.arange(0, length)
        one_hot_frame = F.one_hot(frame_indices, num_classes=length).float()
        one_hot_frame = one_hot_frame.view(1, length, 1, length)
        self.register_buffer("f", one_hot_frame)

        self.embed_frame = Embed([length, 64, 256])

        self.tcn1 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.tcn2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        z = z + self.embed_frame(self.f)
        z = reduce(z, "n t v c -> n t c", "max")  # spatial max pooling

        x = rearrange(z, "n t c -> n c t")
        x = self.tcn1(x)
        x = F.dropout(x, 0.2)
        x = self.tcn2(x)

        x = reduce(x, "n c t -> n c", "max")  # temporal max pooling

        return x


class Embed(nn.Module):
    def __init__(self, feature_dims: List[int] = [64, 64, 64]):
        super().__init__()

        modules = []
        for in_channels, out_channels in zip(feature_dims[:-1], feature_dims[1:]):
            module = self.build_fc(in_channels, out_channels)
            modules.append(module)

        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)

    def build_fc(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
        )


class BatchNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x):
        orig_size = x.size()
        z = self.batch_norm(x.flatten(0, -2))
        return z.view(orig_size)


class AdjacencyMatrix(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.theta = nn.Linear(in_features, out_features)
        self.phi = nn.Linear(in_features, out_features)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        theta = self.theta(z)
        phi = self.phi(z)

        phi = rearrange(phi, "n v c -> n c v")
        S = torch.matmul(theta, phi)
        # S, n v v
        G = self.softmax(S)
        # G, n v v
        return G


class ResidualGCN(nn.Module):
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.lin1 = nn.Linear(in_channels, out_channels)

        self.lin2 = nn.Linear(in_channels, out_channels)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.lin1.weight)
        nn.init.xavier_normal_(self.lin2.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj, mask=None):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = self.lin1(x)

        out = torch.matmul(adj, out)

        out = out + self.lin2(x)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, " f"{self.out_channels})"
