# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import numpy as np

writer = SummaryWriter("logs")

# writer.add_image()

for epoch in range(100):
    writer.add_scalar("y=x", np.random.rand(), epoch)

writer.close()
