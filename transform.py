from PIL import Image
from torchvision_transform import transforms

# python的用法 => tensor数据类型
# 通过transform.ToTensor去解决两个问题
# 1 transform该如何使用
# 2 Tensor数据类型和其他数据类型有什么区别，为什么需要tensor

img_path = "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)

# 将图像转换成tensor
# 选择一个我们需要的工具创建
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)
