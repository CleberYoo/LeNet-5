import argparse
import torch
import torch.nn as nn
import torchvision
import  torchvision.transforms as transforms
from lenet5_pytorch import LeNet5_ver1
import matplotlib.pyplot as plt

def parse_args():
    """
    Parsing the arguments
    """
    parser = argparse.ArgumentParser(prog="LeNet-5")
    parser.add_argument('--batch_size', default=64, type=int, dest='batch_size', help='set the batch size')
    parser.add_argument('--num_epochs', default=10, type=int, dest='number_of_epochs', help='set the number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float, dest='learning_rate', help='set the learning rate')
    parser.add_argument('--num_classes', default=10, type=int, dest='number_of_classes', help='set the number of classes')
    parser.add_argument('--plot', default=False, type=bool, dest='plot', help='whether to draw a diagram')
    parser.add_argument('--savefig', default=False, type=bool, dest='savefig', help='wheter to save the diagram')

    arguments = parser.parse_args()

    return arguments


arguments = parse_args()        # declaration of parser objects

device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
print(f"device : {device}")

# Loading the dataset and preprocessing
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.Compose([transforms.Resize(size=(32,32)), transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]), download=False)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.Compose([transforms.Resize(size=(32,32)), transforms.ToTensor(), transforms.Normalize(mean=(0.1325,), std=(0.3105,))]), download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=arguments.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=arguments.batch_size, shuffle=True)


# # Model
# class ConvNeuralNet(nn.Module):
#     def __init__(self, number_of_classes):
#         super(ConvNeuralNet, self).__init__()
#         self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
#                                     nn.BatchNorm2d(num_features=6),     # original paper didn't write about BN
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
#                                     nn.BatchNorm2d(num_features=16),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc = nn.Linear(in_features=400, out_features=120)
#         self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(in_features=120, out_features=84)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(in_features=84, out_features=number_of_classes)     # number_of_classes = 10

#     def forward(self, image):
#         out = self.layer1(image)
#         out = self.layer2(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         out = self.relu(out)
#         out = self.fc1(out)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         return out


model = LeNet5_ver1(number_of_classes=arguments.number_of_classes).to(device=device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=arguments.learning_rate)      # lr = 1e-3


steps = len(train_loader)       # let's see what is len(train_loader) = 938

loss_list = []

for epoch in range(arguments.number_of_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)        # combine beneath and this line next time
        labels = labels.to(device)

        # Forward
        outputs = model.forward(images)
        loss = loss_function(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 20 == 0:
            loss_list.append(loss.item())
            print(f'Epoch : [{epoch+1} / {arguments.number_of_epochs}], Step : [{i+1} / {steps}], Loss : {loss.item():.4f}')


if arguments.plot == True:
    plt.plot(loss_list)
    plt.xlabel(xlabel='epoch')
    plt.ylabel(ylabel='Loss')
    plt.xticks(ticks=list(range(0, len(loss_list) + 1, 100)))
    plt.title(label='LeNet-5 PyTorch optimizer=Adam')
    if arguments.plot == True:
        plt.savefig('experiment_result/LeNet-5_pytorch:optimizer=Adam.png')
    plt.show()