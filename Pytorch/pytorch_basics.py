import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# Create tensors
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build computational graph (Formulate)
y = w * x + b # y=2x + 3

# Compute gradient
y.backward() # 

# Print gradients (WHY??)
print(x.grad) # x.grad = 2
print(w.grad) # w.grad = 1
print(b.grad) # b.grad = 1

# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #

# Create tensors of shape (10, 3) and (10, 2).
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Build fully connected layer
linear = nn.Linear(3,2)
print('w: ', linear.weight)
print('b: ', linear.bias)

# Build loss function (compute gradient) and optimizer(update weights and bias)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass
pred = linear(x)

# Compute loss (error)
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward pass (Compute gradient)
loss.backward()
print('dL/dw: ', linear.weight.grad)
print('dL/db', linear.bias.grad)

# Update network parameters (1-step gradient descent)
optimizer.step()

# Print out the loss after 1-step gradient descent
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())

# ================================================================== #
#                5. Input pipeline for custom dataset                 #
# ================================================================== #
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 1. Initialize file paths or a list of file names. 
        pass

    def __getitem__(self, index):
        '''
        Defind the iterator
        1) Read on data from file (e.g np.fromfile, PIL.Image.open)
        2) Preprocess the data (torchvision.Transform)
        3) Return data pair (e.g image and label)
        '''
        pass

    def __len__(self):
        return 0 # `0` -> total size of dataset

custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64,
                                           shuffle=True)

# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of model
for param in resnet.parameters():
    param.requires_grad = False

# Replace  the top layer for finetuning
resnet.fc = nn.Linear(resnet.fc.in_features, 100) # input: same as resnet, output: 100 classes for e.g

# Forward pass
images = torch.rand(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size()) # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))