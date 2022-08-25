import torch


class RandomRotate:
    def __init__(self, degrees, axis=0):
        self.degrees = degrees

    def __call__(self, x):
        theta_x, theta_y, theta_z = torch.deg2rad(
            torch.randint(low=-self.degrees, high=self.degrees, size=(3,))
        )

        x = self.rotate_x(x, theta_x)
        x = self.rotate_y(x, theta_y)
        x = self.rotate_z(x, theta_z)

        return x

    def rotate_x(self, x, theta):
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        R_x = torch.tensor(
            [
                [1, 0, 0],
                [0, cos, -sin],
                [0, sin, cos],
            ]
        )

        return self.matmul(R_x, x)

    def matmul(self, R, x):
        return torch.einsum("ij,mtvj->mtvi", R, x)

    def rotate_y(self, x, theta):
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        R_y = torch.tensor(
            [
                [cos, 0, sin],
                [0, 1, 0],
                [-sin, 0, cos],
            ]
        )

        return self.matmul(R_y, x)

    def rotate_z(self, x, theta):
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        R_z = torch.tensor(
            [
                [cos, -sin, 0],
                [sin, cos, 0],
                [0, 0, 1],
            ]
        )

        return self.matmul(R_z, x)
