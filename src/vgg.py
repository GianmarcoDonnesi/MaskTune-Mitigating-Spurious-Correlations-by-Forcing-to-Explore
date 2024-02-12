import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


# Building the Base Model (A reduced version of VGG)
# Defining a New Model with Access to an Intermediate Layer:
# First, we will define a modified version of the model that allows us 
# to access the activations of an intermediate layer, which we will use as our "attention map".

class VGGWithAttention(nn.Module):
    def __init__(self):
        super(VGGWithAttention, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Input: 3 channels, Output: 32 channels, Kernel: 3x3
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)  # Kernel: 2x2
         # Convolutional layer 2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        # # Convolutional layer 3 (reduced compared to standard VGG)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
       # Reduced fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Reduced da 4096
        self.fc2 = nn.Linear(512, 256)  # Reduced da 4096 a 256
        self.fc3 = nn.Linear(256, 10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        # Applying convolutional layers and pooling
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        # Consider activations here as attention map
        attention_map = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.pool(attention_map)
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output, attention_map


# Function for implementing Masking

def apply_attention_masking(inputs, model, attenuation_factor=0.5, dynamic_threshold=True):
    _, attention_maps = model(inputs)

    if dynamic_threshold:
        threshold = attention_maps.quantile(0.75)
    else:
        threshold = attention_maps.mean()

    attention_maps_normalized = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min())

    # Upsample the normalized attention maps to match the input dimensions
    size = (inputs.size(2), inputs.size(3))  # Spatial dimensions of the original input
    attention_maps_upsampled = F.interpolate(attention_maps_normalized, size=size, mode='bilinear', align_corners=False)

    # Calculate the mask using the attenuation factor
    masks = (attention_maps_upsampled > threshold).float() * attenuation_factor

    # Reduce the mask's channel dimensions to 1 for correct broadcasting
    masks = masks.mean(dim=1, keepdim=True)

    # Apply the attenuated mask to the inputs
    masked_inputs = inputs * (1 - masks)

    return masked_inputs


def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = (correct / total) * 100
    return accuracy

def train_model(model, trainloader, criterion, optimizer, num_epochs, device, apply_mask):
    for epoch in range(num_epochs):   # Loop over the dataset for a fixed number of epochs
        running_loss = 0.0
        total_accuracy = 0.0

        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Apply attention masking if requested
            if apply_mask:
                inputs = apply_attention_masking(inputs, model)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass with masked inputs
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print training statistics
            running_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)
            if i % 2000 == 1999:  # Print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f accuracy: %.2f%%' %
                      (epoch + 1, i + 1, running_loss / 2000, total_accuracy / 2000))
                running_loss = 0.0
                total_accuracy = 0.0



if __name__ == '__main__':
    
    # Definition of data transformations for preprocessing the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch Tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
    ])

    # Loading the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("Dataset CIFAR-10 pronto per l'uso.")

    # Setting hyperparameters
    learning_rate = 0.001
    momentum = 0.9
    batch_size = 4
    num_epochs = 12

    # Set the device to 'mps' to use Metal Performance Shaders on Mac
    """MPS is an advanced framework by Apple that provides a wide array of image processing and scientific computing shaders optimized 
    to leverage the GPU (Graphics Processing Unit) hardware of Mac devices."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


    # Model initialization and moving to the appropriate device
    model = VGGWithAttention().to(device)

    # Definition of the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Loading data with specified batch sizes
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)  
    
    # Call to the training function
    # Initial training without masking
    print("Addestramento iniziale senza masking...")
    train_model(model, trainloader, criterion, optimizer, num_epochs=num_epochs, device=device, apply_mask=False)

    # Saving the trained model
    torch.save(model.state_dict(), './vgg_initial.pth')
    print("\nModello iniziale addestrato e salvato.")

    # Loading the trained model
    model.load_state_dict(torch.load('./vgg_initial.pth'))
    print("\nModello iniziale caricato per il fine-tuning.")

    # Optimizer configuration for fine-tuning
    # It's recommended to use a lower learning rate for fine-tuning
    optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # Fine-tuning with masking for a single epoch
    print("\nInizio fine-tuning con masking...")
    train_model(model, trainloader, criterion, optimizer_ft, num_epochs=1, device=device, apply_mask=True)

    # Saving the fine-tuned model
    torch.save(model.state_dict(), './vgg_finetuned.pth')
    print("\nFine-tuning completato e modello salvato.")