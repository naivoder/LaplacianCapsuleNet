import torch
import torch.nn as nn
import unittest


class PyramidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_dim):
        super(PyramidBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([out_channels, input_dim, input_dim])
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([out_channels, input_dim // 2, input_dim // 2])
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = torch.relu(self.ln1(self.conv1(x)))
        # x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.ln2(self.conv2(x)))
        # x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        return x


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-9)


class Length(nn.Module):
    def forward(self, inputs):
        return torch.sqrt((inputs ** 2).sum(dim=-1))


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsule=16, dim_vector=8, num_routing=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing

    def forward(self, x):
        batch_size = x.size(0)
        u_hat = x.view(batch_size, -1, 1, self.dim_vector)
        b_ij = torch.zeros(1, u_hat.size(1), self.num_capsule, 1).to(x.device)
        for i in range(self.num_routing):
            c_ij = torch.nn.functional.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)
            if i < self.num_routing - 1:
                b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)
        return v_j.squeeze()

class LaplacianCapsuleNet(nn.Module):
    def __init__(self, input_shape, num_classes, num_routing=3):
        super(LaplacianNet, self).__init__()
        self.pyramid_block1 = PyramidBlock(input_shape[0], 64, 128)
        self.pyramid_block2 = PyramidBlock(64 + input_shape[0], 128, 64)
        self.pyramid_block3 = PyramidBlock(128 + input_shape[0], 256, 32)

        self.primary_caps = nn.Conv2d(256 + input_shape[0], 32 * 8, kernel_size=11, stride=1, padding=0)
        self.primary_caps_reshape = nn.Sequential(
            nn.Conv2d(32 * 8, 32 * 8, kernel_size=1),
            nn.BatchNorm2d(32 * 8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 6 * 6, 32 * 8)  # Adjusted linear layer input size
        )
        self.capsule_layer = CapsuleLayer(num_capsule=num_classes, dim_vector=8, num_routing=num_routing)
        self.length = Length()

    def forward(self, inputs, y=None):
        x_l1, x_l2, x_l3, x_l4 = inputs

        x1 = self.pyramid_block1(x_l1)
        x2 = self.pyramid_block2(torch.cat((x1, x_l2), dim=1))
        x3 = self.pyramid_block3(torch.cat((x2, x_l3), dim=1))
        x = torch.cat([x3, x_l4], dim=1)

        x = self.primary_caps(x)
        x = self.primary_caps_reshape(x)
        x = self.capsule_layer(x)

        length = self.length(x)
        return length
    
class LaplacianNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LaplacianNet, self).__init__()
        self.fc_input = (256 + input_shape[0]) * 16 * 16
        self.pyramid_block1 = PyramidBlock(input_shape[0], 64, 128)
        self.pyramid_block2 = PyramidBlock(64 + input_shape[0], 128, 64)
        self.pyramid_block3 = PyramidBlock(128 + input_shape[0], 256, 32)
        self.fc = nn.Linear(self.fc_input, 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, inputs, y=None):
        x_l1, x_l2, x_l3, x_l4 = inputs

        x1 = self.pyramid_block1(x_l1)
        x2 = self.pyramid_block2(torch.cat((x1, x_l2), dim=1))
        x3 = self.pyramid_block3(torch.cat((x2, x_l3), dim=1))
        x = torch.cat([x3, x_l4], dim=1)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = nn.functional.relu(self.fc(x))
        return self.output(x)


def margin_loss(y_true, y_pred):
    y_true = torch.nn.functional.one_hot(y_true, num_classes=y_pred.size(1)).float()
    alpha_margin = torch.clamp(0.9 - y_pred, min=0.0) ** 2
    beta_margin = torch.clamp(y_pred - 0.1, min=0.0) ** 2
    L = y_true * alpha_margin + 0.5 * (1 - y_true) * beta_margin
    return L.mean()



class TestPyramidBlock(unittest.TestCase):
    def test_forward(self):
        batch_size = 4
        in_channels = 3
        out_channels = 64
        input_shape = (128, 128)

        x = torch.randn(batch_size, in_channels, *input_shape)
        pyramid_block = PyramidBlock(in_channels, out_channels, input_shape[0])
        output = pyramid_block(x)

        self.assertEqual(output.shape, (batch_size, out_channels, input_shape[0] // 2, input_shape[1] // 2))

class TestCapsuleLayer(unittest.TestCase):
    def test_forward(self):
        batch_size = 4
        num_capsule = 16
        dim_vector = 8
        input_dim = 64

        x = torch.randn(batch_size, input_dim)
        capsule_layer = CapsuleLayer(num_capsule=num_capsule, dim_vector=dim_vector, num_routing=3)
        output = capsule_layer(x)

        self.assertEqual(output.shape, (batch_size, num_capsule, dim_vector))

class TestLength(unittest.TestCase):
    def test_forward(self):
        batch_size = 4
        num_capsule = 16
        dim_vector = 8

        inputs = torch.randn(batch_size, num_capsule, dim_vector)
        length_layer = Length()
        output = length_layer(inputs)

        self.assertEqual(output.shape, (batch_size, num_capsule))

class TestLaplacianNet(unittest.TestCase):
    def test_forward(self):
        batch_size = 2
        input_shape = (3, 128, 128)
        num_classes = 10

        x_l1 = torch.randn(batch_size, *input_shape)
        x_l2 = torch.randn(batch_size, 3, 64, 64)
        x_l3 = torch.randn(batch_size, 3, 32, 32)
        x_l4 = torch.randn(batch_size, 3, 16, 16)
        y = torch.randint(0, num_classes, (batch_size,))

        model = LaplacianNet(input_shape, num_classes)
        length = model([x_l1, x_l2, x_l3, x_l4], y)

        self.assertEqual(length.shape, (batch_size, num_classes))


if __name__ == '__main__':
    unittest.main()