import torchvision
from tensorboardX import SummaryWriter

dataset_tansform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True,
                                         transform=dataset_tansform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                        transform=dataset_tansform, download=True)

# print(train_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(f"img is {img},target is {target},class is {test_set.classes[target]}")

print(test_set[0])

writer = SummaryWriter("p10")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("test_image", img, i)

writer.close()
