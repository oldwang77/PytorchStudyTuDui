from torch.utils.data import Dataset
from PIL import Image
import os


# 继承DataSet类
class MyData(Dataset):

    # 提供全局变量，为后面的方法提供变量
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir

        # label的位置
        self.path = os.path.join(self.root_dir, self.label_dir)
        # image的位置
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        # 拿到了每个图片的地址
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)

        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    root_dir = "hymenoptera_data/train"
    ant_label_dir = "ants"
    bee_label_dir = "bees"

    ants_dataset = MyData(root_dir=root_dir, label_dir=ant_label_dir)
    bee_dataset = MyData(root_dir=root_dir,label_dir=bee_label_dir)

    img, label = ants_dataset[0]
    print(f"img is {img},label is {label}")

    # 我们整个训练集
    train_dataset = ants_dataset + bee_dataset
    print(train_dataset.__getitem__(0))