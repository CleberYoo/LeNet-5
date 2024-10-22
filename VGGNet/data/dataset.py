import os
from glob import glob
import torch
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader


class STL10(torch.utils.data.Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        # classPath = self.root + 'class_names.txt'
        # classPath = os.path.join(self.root, 'class_names.txt')
        classPath = os.path.join(self.root, 'class_names.txt')
        
        self.labels = self.readLabel(filePath=classPath)

        self.train_loader, self.test_loader = self.train_test()


    def readLabel(self, filePath:str):
        with open(file=filePath, mode='r') as file:
            label_list = []
            for line in file:
                label_list.append(line.rstrip())
        
        print(f"classes : {label_list}")
        return label_list


    def getMean_Std(self):
        imagePath = os.path.join(self.root, 'image')
        transform = transforms.Compose(transforms=[transforms.ToTensor()])
        dataset = datasets.ImageFolder(root=imagePath, transform=transform)

        meanRGB = []
        stdRGB = []
        for x, _ in dataset:
            meanRGB.append(np.mean(a=x.numpy(), axis=(1, 2)))
            stdRGB.append(np.mean(a=x.numpy(), axis=(1, 2)))

        meanR = np.mean(a=[m[0] for m in meanRGB])
        meanG = np.mean(a=[m[1] for m in meanRGB])
        meanB = np.mean(a=[m[2] for m in meanRGB])

        stdR = np.mean(a=[s[0] for s in stdRGB])
        stdG = np.mean(a=[s[1] for s in stdRGB])
        stdB = np.mean(a=[s[2] for s in stdRGB])


        print(f"mean : {meanR, meanG, meanB}")
        print(f"std : {stdR, stdG, stdB}")

        return meanR, meanG, meanB, stdR, stdG, stdB


    def train_test(self):
        imagePath = os.path.join(self.root, 'image')
        meanR, meanG, meanB, stdR, stdG, stdB = self.getMean_Std()
        # stdR, stdG, stdB = self.getStd()

        transform = transforms.Compose(transforms=[transforms.Resize(size=256),
                                                #    transforms.FiveCrop(size=224),
                                                   transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                                   transforms.Normalize(mean=[meanR, meanG, meanB],
                                                                        std=[stdR, stdG, stdB])
                                                    ])
        dataset = datasets.ImageFolder(root=imagePath, transform=transform)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[train_size, test_size])

        train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=2)
        test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, num_workers=2)

        return train_loader, test_loader


if __name__ == "__main__":
    stl10Path = '/home/msis/Dataset/stl10'
    STL10_ = STL10(root=stl10Path)

    train_dataloader, test_dataloader = STL10_.train_loader, STL10_.test_loader