import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import unittest

class ImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label

class ImagePyramidDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def upscale(self, img, target_size):
        return transforms.functional.resize(img, target_size, antialias=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        g1 = img
        g2 = transforms.Resize((64, 64))(g1)
        g3 = transforms.Resize((32, 32))(g1)
        g4 = transforms.Resize((16, 16))(g1)

        # Laplacian pyramid differences
        l1 = g1
        l2 = torch.abs(g1 - self.upscale(g2, g1.shape[1:]))
        l3 = torch.abs(g2 - self.upscale(g3, g2.shape[1:]))
        l4 = torch.abs(g3 - self.upscale(g4, g3.shape[1:]))

        # Resize Laplacians to original sizes
        l2 = transforms.Resize((64, 64))(l2)
        l3 = transforms.Resize((32, 32))(l3)
        l4 = transforms.Resize((16, 16))(l4)

        return [l1, l2, l3, l4], label

def plot_laplacian_pyramid(image_dataset):
    pyramid, _ = image_dataset[100]

    # Plot the Laplacian pyramid
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    for i, lap in enumerate(pyramid):
        img = lap.permute(1, 2, 0).numpy()
        img = (img - img.min())/(img.max() - img.min())
        axs[i].imshow(img)
        axs[i].set_title(f'Laplacian Level {i + 1}')
        axs[i].axis('off')
    plt.savefig("results/sample_laplacian_pyramid.png")
    plt.show()

class TestImagePyramidDataset(unittest.TestCase):
    def setUp(self):
        # Create small dummy datasets for testing
        self.transform_gray = transforms.Compose([
            transforms.Grayscale(),  # Ensure single channel
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform_rgb = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.dataset_gray = torchvision.datasets.FakeData(transform=self.transform_gray)
        self.dataset_rgb = torchvision.datasets.FakeData(transform=self.transform_rgb)
        
    def test_image_pyramid_gray(self):
        dataset_gray = ImagePyramidDataset(self.dataset_gray)
        dataloader_gray = DataLoader(dataset_gray, batch_size=2, shuffle=True)
        
        for inputs, _ in dataloader_gray:
            self.assertEqual(len(inputs), 4)
            self.assertEqual(inputs[0].shape, torch.Size([2, 1, 128, 128]))
            self.assertEqual(inputs[1].shape, torch.Size([2, 1, 64, 64]))
            self.assertEqual(inputs[2].shape, torch.Size([2, 1, 32, 32]))
            self.assertEqual(inputs[3].shape, torch.Size([2, 1, 16, 16]))
            break
        
    def test_image_pyramid_rgb(self):
        dataset_rgb = ImagePyramidDataset(self.dataset_rgb)
        dataloader_rgb = DataLoader(dataset_rgb, batch_size=2, shuffle=True)
        
        for inputs, _ in dataloader_rgb:
            self.assertEqual(len(inputs), 4)
            self.assertEqual(inputs[0].shape, torch.Size([2, 3, 128, 128]))
            self.assertEqual(inputs[1].shape, torch.Size([2, 3, 64, 64]))
            self.assertEqual(inputs[2].shape, torch.Size([2, 3, 32, 32]))
            self.assertEqual(inputs[3].shape, torch.Size([2, 3, 16, 16]))
            break

if __name__ == '__main__':
    # Run unit tests
    unittest.main(exit=False)

    # Initialize CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=False, transform=transform)
    image_pyramid_dataset = ImagePyramidDataset(cifar_dataset)

    # Plot Laplacian pyramid for a random animal image from CIFAR-10
    plot_laplacian_pyramid(image_pyramid_dataset)
