import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

dataset_tansform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                         transform=dataset_tansform, download=True)

# comand+p显示提示
test_loader = DataLoader(dataset=test_data, batch_size=64,
                         shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("dataloader")
step = 0
# 测试数据集中第一张图片集和target
for data in test_loader:
    imgs, targets = data
    # print(img.shape)
    # print(target)
    writer.add_images("test_data", imgs, step)
    step = step + 1

writer.close()

#  tensorboard --logdir=dataloader --port=6007
