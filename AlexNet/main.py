import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import progress
from CNN import AlexNet
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="AlexNet")
    parser.add_argument("--epochs", dest="epochs", help="set the epochs", default=10, type=int)
    parser.add_argument("--batch_size", dest="batch_size", help="set the batch size", default=512, type=int)
    parser.add_argument("--image_size", dest="image_size", help="set the length of image", default=227, type=int)

    args = parser.parse_args()

    return args

args = parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"         # let's make this upper case next time

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                # total 10 classes

print(f"{device} is ready")
print(f"total GPU : {torch.cuda.device_count()}")
print(f"current device : {torch.cuda.current_device()}")



# Data preprocessing
transform = transforms.Compose(transforms=[transforms.Resize(size=(227, 227)),
                                           transforms.ToTensor()
                                           ]
                               )        # this object help transform about image

train_data = datasets.FashionMNIST(root="data", train=True, transform=transform, download=False)
validation_data = datasets.FashionMNIST(root="data", train=False, transform=transform, download=False)

train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_data, batch_size=args.batch_size, shuffle=True)



# Model declaration
model = AlexNet().to(device=device)
criterion = F.nll_loss        # nll_loss() : negative log likelihood loss
optimizer = optim.Adam(params=model.parameters())


# data_iter = iter(train_loader)
# data, target = next(data_iter)

summary(model=model, input_size=(1, 227, 227), batch_size=args.batch_size)



# train
for epoch in range(1, args.epochs+1):
    progress.train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch, criterion=criterion)
    progress.test(model=model, device=device, test_loader=validation_loader, criterion=criterion)
    
    print("HERE")