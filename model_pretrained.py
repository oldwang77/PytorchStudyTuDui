import torch
import torchvision

# train_data = torchvision.datasets.ImageNet("./data_image_net",split='train',download=True,
#                                            transform=torchvision.transforms.ToTensor())

# vgg16_false = torchvision.models.vgg16(pretrained=False)
from torch import nn

vgg16_true = torchvision.models.vgg16(pretrained=True)
vgg16_false = torchvision.models.vgg16(pretrained=False)
print("ok")

print(vgg16_true)

# 看vgg的结构我们知道，它的最后一层是线性层，
# 分成了1000类，我们可以通过加一层，分成10类
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 如果我们想再seq里面加
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 如果我们不是增加，是为了修改
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)