import torch


def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out


from torch import nn  # 神经网络相关工作
from torch.nn import functional as F  # 常用函数
from torch import optim  # 优化工具包

import torchvision

# Set the value of batchsize
batch_size = 2048
# Get loader with torchvision.datasets
# Convert numpy to tensor
# Normalized the data
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])), batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])), batch_size=batch_size, shuffle=False)

import vit

# Use CUDA for speed
device = torch.device('cuda')
# Take net into CUDA
net = vit.VisionTransformer().to(device)
# Use Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.003)
train_loss = []
# Train
for epoch in range(10):
    # Get data from dataloader
    for batch_idx, (x, y) in enumerate(train_loader):
        # Take images into CUDA
        x = x.to(device)
        # Get output of net
        out = net(x)
        # Change the format of labels to one-hot
        y_onehot = one_hot(y).to(device)
        # Get loss
        loss = F.mse_loss(out, y_onehot)
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Store loss
        train_loss.append(loss.item())
        # Show loss
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

# Test
total_correct = 0
for x, y in test_loader:
    # Take images and labels into CUDA
    x = x.to(device)
    y = y.to(device)
    # Get prediction from network
    out = net(x)
    pred = out.argmax(dim=1)
    # Calculate result
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

