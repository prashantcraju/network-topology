import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Fully Connected Network (FCN)
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to evaluate the model and calculate accuracy
def evaluate_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Function to perform lesioning (lobotomy) by zeroing out certain neurons
def lesion_neurons(model, layer_name, neuron_indices, testloader):
    original_state = {}
    
    # Save the original weights or activations to revert later
    for name, param in model.named_parameters():
        if layer_name in name:
            original_state[name] = param.data.clone()

            # Set the neurons to zero (lesion)
            param.data[:, neuron_indices] = 0

    # Re-evaluate the model after lesioning
    lesioned_accuracy = evaluate_model(model, testloader)
    
    # Restore the original state
    for name, param in model.named_parameters():
        if name in original_state:
            param.data = original_state[name].clone()
    
    return lesioned_accuracy

# Lesioning Example: Disable neurons from the second fully connected layer in FCN
fcn_model = FCN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fcn_model.parameters(), lr=0.001)

# Train the model (using a simple training loop here)
for epoch in range(1):
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = fcn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model accuracy before lesioning
print(f"Initial FCN Accuracy: {evaluate_model(fcn_model, testloader)}%")

# Lesioning the FCN model: Disable neurons in the second fully connected layer
neuron_indices_to_lesion = [10, 20, 30]  # Example: lesioning specific neurons
lesioned_accuracy = lesion_neurons(fcn_model, 'fc2.weight', neuron_indices_to_lesion, testloader)

print(f"Lesioned FCN Accuracy (Neurons {neuron_indices_to_lesion}): {lesioned_accuracy}%")

# Example with CNN model
cnn_model = CNN()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Train the model (using a simple training loop here)
for epoch in range(1):
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model accuracy before lesioning
print(f"Initial CNN Accuracy: {evaluate_model(cnn_model, testloader)}%")

# Lesioning the CNN model: Disable neurons in the first fully connected layer
neuron_indices_to_lesion = [10, 20, 30]  # Example: lesioning specific neurons
lesioned_accuracy = lesion_neurons(cnn_model, 'fc1.weight', neuron_indices_to_lesion, testloader)

print(f"Lesioned CNN Accuracy (Neurons {neuron_indices_to_lesion}): {lesioned_accuracy}%")
