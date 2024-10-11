import os
from torchvision import datasets
import torchvision.transforms as transforms

rootPath = '/home/msis/Work/vggnet/dataset'

if os.path.exists(path=rootPath) == False:
    os.makedirs(name=rootPath, exist_ok=True)

train = datasets.STL10(root=rootPath, split="train", transform=transforms.ToTensor())
test = datasets.STL10(root=rootPath, split="test", transform=transforms.ToTensor())