from torch import nn

from einops import rearrange, parse_shape

from sgn.layers import DynamicsRepresentation, JointLevelModule, FrameLevelModule


class SGN(nn.Module):
    def __init__(self, num_classes, length, num_joints, num_features):
        super().__init__()

        self.num_classes = num_classes
        self.length = length
        self.num_joints = num_joints
        self.num_features = num_features

        self.dynamics_representation = DynamicsRepresentation(num_features, num_joints)

        self.joint_level_module = JointLevelModule(num_joints)
        self.frame_level_module = FrameLevelModule(length)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(-1, self.length, self.num_joints, self.num_features)

        z = self.dynamics_representation(x)
        z = rearrange(z, "n t v c -> (n t) v c", **parse_shape(x, "n t _ _"))
        z = self.joint_level_module(z)
        z = rearrange(z, "(n t) v c -> n t v c", **parse_shape(x, "n _ v _"))
        z = self.frame_level_module(z)
        logits = self.fc(z)
        return logits
