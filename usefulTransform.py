from torchvision_transform import transforms
from PIL import Image

image_path = "hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(image_path)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img_PIL)

# Normalize归一化计算
# output[channel] = (input[channel] - mean[channel]) / std[channel]``
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)

# resize的使用
# (768, 512)
# (512, 512)
print(img_PIL.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img_PIL)
print(img_resize.size)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize2 = trans_compose(img_PIL)
print(f"img_resize2 size is = {img_resize2.size}")

# 随机裁剪RandomCrop
trans_random = transforms.RandomCrop([500, 400])
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
# 裁剪10
for i in range(5):
    img_crop = trans_compose_2(img_PIL)
    print(img_crop)
