import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Download MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
# Fully connected neural network with 1 hidden layer
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size) # Initilize `class Linear(Module):`
    self.relu = nn.ReLU() # Intialize Relu object
    self.fc2 = nn.Linear(hidden_size, num_classes) # Initilize Linear object

  def forward(self, x):
    '''
    def linear(
        input: Tensor,
        weight: Tensor,
        .....
    ) -> Tensor:
    '''
    out = self.fc1(x) # Out: Tensor
    out = self.relu(out) # Out: Tensor
    out = self.fc2(out) # Out: Tensor
    
    return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Intilize Loss and Optimizer
criterion = nn.CrossEntropyLoss() # Intialize Loss object (e.g CrossEntropy, MSE, etc)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Intilize Optimizer object (e.g SGD, Adam, etc)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader): # multiple images because train_loader loads by batches (e.g 32 images)
    # 1) Move tensors to the configured device
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    
    # 2) Forward pass
    # - Compute prediction
    preds = model(images)
    # - Compute Loss
    loss = criterion(preds, labels)

    # 3) Backward (Compute Gradient) and optimize 
    # Compute gradient via backpropagation
    optimizer.zero_grad() # Reset the gradient, only use gradient on current batch
    loss.backward()
    # Update model parameters
    optimizer.step()

    if (i+100) % 100 == 0: # Every 100 batches
      print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')