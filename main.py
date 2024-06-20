from train import train_and_evaluate
import torchvision 
import torchvision.transforms as transforms
import torch 
import os

os.makedirs('results', exist_ok=True)
os.makedirs('weights', exist_ok=True)

print('🕺 lets gooooooooooooooooooooooooooo' if torch.cuda.is_available() else '💩 f@*# the cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_datasets = {
    'CIFAR10': torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform),
    'CIFAR100': torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform),
    'MNIST': torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform),
    'FashionMNIST': torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform),
    'SVHN': torchvision.datasets.SVHN(root='../data', split='train', download=True, transform=transform),
    'STL10': torchvision.datasets.STL10(root='../data', split='train', download=True, transform=transform),
    # 'ImageNet': torchvision.datasets.ImageNet(root='../data', split='train', download=True, transform=transform),
    'Caltech101': torchvision.datasets.Caltech101(root='../data', download=True, transform=transform),
    'Caltech256': torchvision.datasets.Caltech256(root='../data', download=True, transform=transform),
    'CelebA': torchvision.datasets.CelebA(root='../data', split='train', download=True, transform=transform),
    # 'LSUN': torchvision.datasets.LSUN(root='../data', classes='train', transform=transform,),
    'Omniglot': torchvision.datasets.Omniglot(root='../data', download=True, transform=transform),
    'OxfordIIITPet': torchvision.datasets.OxfordIIITPet(root='../data', split='trainval', download=True, transform=transform),
    # 'StanfordCars': torchvision.datasets.StanfordCars(root='../data', split='train', download=True, transform=transform),
    'SBD': torchvision.datasets.SBDataset(root='../data', image_set='train', download=False)
}

test_datasets = {
    'CIFAR10': torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform),
    'CIFAR100': torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform),
    'MNIST': torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform),
    'FashionMNIST': torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform),
    'SVHN': torchvision.datasets.SVHN(root='../data', split='test', train=False, download=True, transform=transform),
    'STL10': torchvision.datasets.STL10(root='../data', split='test', train=False, download=True, transform=transform),
    # 'ImageNet': torchvision.datasets.ImageNet(root='../data', split='test', download=True, transform=transform),
    'Caltech101': torchvision.datasets.Caltech101(root='../data', download=False, train=False, transform=transform),
    'Caltech256': torchvision.datasets.Caltech256(root='../data', download=False, train=False, transform=transform),
    'CelebA': torchvision.datasets.CelebA(root='../data', split='test', download=True, transform=transform),
    # 'LSUN': torchvision.datasets.LSUN(root='../data', classes='test', transform=transform,),
    'Omniglot': torchvision.datasets.Omniglot(root='../data', download=False, train=False, transform=transform),
    'OxfordIIITPet': torchvision.datasets.OxfordIIITPet(root='../data', split='test', download=True, transform=transform),
    # 'StanfordCars': torchvision.datasets.StanfordCars(root='../data', split='train', download=True, transform=transform),
    'SBD': torchvision.datasets.SBDataset(root='../data', image_set='val', download=False)
}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

for name in train_datasets.keys():
    print(f"\nTraining on {name} dataset")
    # test_data = torchvision.datasets.__dict__[name](root='./data', train=False, download=True, transform=transform)
    
    model = train_and_evaluate(name, train_datasets[name], test_datasets[name], device)