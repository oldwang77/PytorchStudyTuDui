import torch
import torchvision

from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset_tansform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                       transform=dataset_tansform, download=True)
dataLoader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()
print(tudui)
for data in dataLoader:
    imgs, targets = data
    output = tudui(imgs)
    # print(output)

    # torch.Size([64,6,30,30]) = > ([xxx,3,30,30])
    # 当不知道xxx是多少的时候，直接写-1，会自己帮我们计算的
    torch.reshape(output, (-1, 3, 30, 30))
