import torch
import torch.nn

# 模型保存的方式1和2，见model_save

# 方式1 保存vgg16_method1.pth
import torchvision
from torch import nn

model = torch.load("vgg16_method1.pth")
print(model)

# 方式2 加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(model)


# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)


tudui = Tudui()
model = torch.save(tudui, 'tudui_method1.pth')

# model = torch.load('tudui_method1.pth')
# print(model)
