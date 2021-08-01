import torch
from torch import nn
from torch.nn import L1Loss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction="sum")
result = loss(inputs, targets)

# tensor(2.)
# tensor(1.3333)
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

print(result)
print(result_mse)

