import torch

def print_hi(name):
    print(f'Hi, {name}')
    print(torch.cuda.is_available())
    # 我们可以查看torch里面的包
    # dir(torch)

    # help给我们介绍它的语法
    # help(torch.cuda.is_available)


if __name__ == '__main__':
    print_hi('PyCharm')


