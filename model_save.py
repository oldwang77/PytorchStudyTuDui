import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式
# 模型的结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2
# 保存网络模型成字典的形式
# 保存模型的参数
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
