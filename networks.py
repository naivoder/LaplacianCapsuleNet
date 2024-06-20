import torch
import torch.nn as nn
import unittest

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_dim):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([out_channels, input_dim, input_dim])
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([out_channels, input_dim // 2, input_dim // 2])
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = torch.relu(self.ln1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.ln2(self.conv2(x)))
        x = self.dropout(x)
        return x

class CNNNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNNet, self).__init__()
        self.fc_input = 256 * 16 * 16
        self.block1 = CNNBlock(input_shape[0], 64, 128)
        self.block2 = CNNBlock(64, 128, 64)
        self.block3 = CNNBlock(128, 256, 32)
        self.fc = nn.Linear(self.fc_input, 128)
        self.output = nn.Linear(128, num_classes)
        print('Using CNNNet')

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc(x))
        return self.output(x)

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
        x = self.pool(x)
        x = torch.relu(self.ln2(self.conv2(x)))
        x = self.dropout(x)
        return x
    
class LaplacianNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LaplacianNet, self).__init__()
        self.fc_input = (256 + input_shape[0]) * 16 * 16
        self.pyramid_block1 = PyramidBlock(input_shape[0], 64, 128)
        self.pyramid_block2 = PyramidBlock(64 + input_shape[0], 128, 64)
        self.pyramid_block3 = PyramidBlock(128 + input_shape[0], 256, 32)
        self.fc = nn.Linear(self.fc_input, 128)
        self.output = nn.Linear(128, num_classes)
        print('Using LaplacianNet')

    def forward(self, inputs):
        x_l1, x_l2, x_l3, x_l4 = inputs

        x1 = self.pyramid_block1(x_l1)
        x2 = self.pyramid_block2(torch.cat((x1, x_l2), dim=1))
        x3 = self.pyramid_block3(torch.cat((x2, x_l3), dim=1))
        x = torch.cat([x3, x_l4], dim=1)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc(x))
        return self.output(x)

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