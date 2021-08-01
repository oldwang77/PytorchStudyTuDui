import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset_tansform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                         transform=dataset_tansform, download=True)

dataloader = DataLoader(dataset=test_data, batch_size=1)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
tudui = Tudui()
# 优化器
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
for epoch in range(20):
    run_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        # target告诉了类别信息
        result_loss = loss(outputs, targets)
        # 梯度清0
        optim.zero_grad()
        # 求出每个结点的梯度
        result_loss.backward()
        # 对参数进行调优
        optim.step()
        run_loss += result_loss
    print(f"the epoch {epoch} loss is {result_loss}")