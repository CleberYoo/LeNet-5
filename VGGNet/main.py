import sys
import torch
from NeuralNetwork import VGGNet
from util.constants import VGG16, VGG19
from torchsummary import summary
from data.dataset import STL10
import torch.nn as nn
from torch import optim
from util.functions import *
from torch.optim.lr_scheduler import StepLR
# import torchvision

print(f"python version : {sys.version}")
print(f"version info : {sys.version_info}")
print(f"cuda available : {torch.cuda.is_available()}")
print(f"pytorch version : {torch.__version__}")
print(f"CUDA version : {torch.version.cuda}")
print(f"number of GPUs : {torch.cuda.device_count()}")
print(f"current CUDA device : {torch.cuda.get_device_name(device=torch.cuda.current_device())}")

stl10Path = '/home/msis/Dataset/stl10/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = VGGNet(model=VGG16, in_channels=3, num_classes=10, init_weights=True).to(device=device)
summary(model=model, input_size=(3, 224, 224), device=device)

STL10 = STL10(root=stl10Path)
train_dataloader, val_dataloader = STL10.train_loader, STL10.test_loader

# for x in train_dataloader:
#     print(x)

loss_func = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(params=model.parameters(), lr=0.01)
current_lr = get_lr(opt=optimizer)
print(f'current lr : {current_lr}')

scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

# define the training parameters
params_train = {'num_epochs'    : 100,
                'optimizer'     : optimizer,
                'loss_func'     : loss_func,
                'train_dl'      : train_dataloader,
                'val_dl'        : val_dataloader,
                'sanity_check'  : False,
                'scheduler'     : scheduler,
                'path2weights'  : './models/weights.pt'}

createFolder(directory='./models')

model, loss_hist, metric_hist = train_val(model=model, params=params_train)