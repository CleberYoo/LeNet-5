import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


batch_size = 64
number_of_classes = 10
learning_rate = 0.001
number_of_epochs = 10

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device : {device}")

# 데이터셋 및 DataLoader 설정
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.Compose([transforms.Resize(size=(32, 32)), transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]), download=False)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.Compose([transforms.Resize(size=(32, 32)), transforms.ToTensor(), transforms.Normalize(mean=(0.1325,), std=(0.3105,))]), download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# 모델 정의
class ConvNeuralNet(nn.Module):
    def __init__(self, number_of_classes):
        super(ConvNeuralNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
                                    nn.BatchNorm2d(num_features=6),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
                                    nn.BatchNorm2d(num_features=16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=400, out_features=120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=84, out_features=number_of_classes)

    def forward(self, image):
        out = self.layer1(image)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


# 실험 함수 정의
def run_experiment(optimizer_type, model, loss_function, train_loader, number_of_epochs):
    optimizer = optimizer_type(params=model.parameters(), lr=learning_rate)
    steps = len(train_loader)
    loss_list = []

    for epoch in range(number_of_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                loss_list.append(loss.item())
                print(f'Optimizer: {optimizer_type.__name__}, Epoch: [{epoch + 1} / {number_of_epochs}], Step: [{i + 1} / {steps}], Loss: {loss.item()}')

    return loss_list


# 실험 실행 및 결과 플로팅
optimizers = [torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop, torch.optim.Adagrad]
fig, axes = plt.subplots(nrows=len(optimizers), figsize=(8, 2 * len(optimizers)), sharex=True)
fig.suptitle('Training Loss over Iterations for Different Optimizers')

for ax, optimizer_type in zip(axes, optimizers):
    model = ConvNeuralNet(number_of_classes=number_of_classes).to(device=device)
    loss_function = nn.CrossEntropyLoss()
    loss_list = run_experiment(optimizer_type, model, loss_function, train_loader, number_of_epochs)
    ax.plot(loss_list)
    ax.set_title(optimizer_type.__name__)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('experiment_result/optimizer_comparison.png')
plt.show()
