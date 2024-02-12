import torch
import torchvision
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support


# Paths for images and attributes
IMG_PATH = './data/celeba/img_align_celeba/img_align_celeba/'
ATTR_PATH = './data/celeba/list_attr_celeba.csv'
PARTITION_PATH = './data/celeba/list_eval_partition.csv'

# Load the dataset attributes
attributes = pd.read_csv(ATTR_PATH)
partitions = pd.read_csv(PARTITION_PATH)
num_classes = 10  # Modify this to the number of attributes you want to predict


# Define the dataset class
class CelebADataset(Dataset):
    def __init__(self, dataset, img_path, partition, transform=None):
        self.img_path = img_path
        self.dataset = dataset[dataset['partition'] == partition].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, self.dataset.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        attrs = self.dataset.iloc[idx, 1:1+num_classes].values.astype('float')

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(attrs)



# Define transformations (e.g., resizing and normalization)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Merge attributes and partitions on the image filename
dataset = attributes.merge(partitions, on='image_id')


# Initialize the datasets
train_dataset = CelebADataset(dataset, IMG_PATH, partition=0, transform=transform)
valid_dataset = CelebADataset(dataset, IMG_PATH, partition=1, transform=transform)
test_dataset = CelebADataset(dataset, IMG_PATH, partition=2, transform=transform)

 # Initialize the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#If you want to train the model on a subset of CelebA
# Define the subset size
subset_size = 1000  # For example, use 1000 images for training

# Generate random indices for the subset
train_indices = torch.randperm(len(train_dataset))[:subset_size]

# Create the subset of the training dataset
train_subset = Subset(train_dataset, train_indices)

# Create the DataLoader for the training dataset subset
train_loader_subset = DataLoader(train_subset, batch_size=32, shuffle=True)


# Definition of AttentionMaskingResNet50 model
class AttentionMaskingResNet50(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(AttentionMaskingResNet50, self).__init__()
        # Use features from the pretrained model, excluding the last layers
        self.features = nn.Sequential(*list(pretrained_model.children())[:-2])
        self.avgpool = pretrained_model.avgpool
        # Adds a mask generator based on the output of the penultimate layer
        self.mask_generator = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1),  # Assuming the feature output has 2048 channels
            nn.Sigmoid()
        )
        # Uses the number of features coming out of avgpool for the fully connected layer
        self.fc_in_features = pretrained_model.fc.in_features
        self.classifier = nn.Linear(self.fc_in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        mask = self.mask_generator(x)
        # Applies the mask directly to the features before pooling
        x = x * mask
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def calculate_metrics(outputs, labels):
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(outputs)
    # Converts probabilities to binary predictions
    preds = (probs > 0.5).float()

    # True Positives, False Positives, True Negatives, False Negatives
    TP = (preds * labels).sum(dim=0)
    FP = ((1 - labels) * preds).sum(dim=0)
    FN = (labels * (1 - preds)).sum(dim=0)
    TN = ((1 - labels) * (1 - preds)).sum(dim=0)

    # Precision, Recall, and F1 for each class, then calculate the average
    precision = (TP / (TP + FP + 1e-8)).mean()
    recall = (TP / (TP + FN + 1e-8)).mean()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1.item()




def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    print('Addestramento iniziato...')
    
    for epoch in range(num_epochs):  # loop over the dataset a fixed number of epochs
        running_loss = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for inputs, labels in train_loader: 
            # Get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass e optimization
            loss.backward()
            optimizer.step()
            
            # Calculate metrics for the current batch
            precision, recall, f1 = calculate_metrics(outputs, labels)
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Print training statistics
            running_loss += loss.item()
            
        # Calculate average metrics for the epoch
        avg_precision = total_precision / len(train_loader)
        avg_recall = total_recall / len(train_loader)
        avg_f1 = total_f1 / len(train_loader)
        
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}')

    print('Addestramento completato')

    

if __name__ == '__main__':
    
    # Setting hyperparameters
    learning_rate = 0.001
    momentum = 0.9
    batch_size = 32
    num_epochs = 20
    
    # Set the device to 'mps' to use Metal Performance Shaders on Mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Initial training without Masking
    
    # Load a pretrained ResNet50 without masking components
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
    resnet50.to(device)
    
    
    # Definition of the loss function for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(resnet50.parameters(), lr=learning_rate, momentum=momentum)
            
    # Call the training function on the entire dataset
    train_model(resnet50, train_loader, criterion, optimizer, num_epochs, device)
    
    PATH1 = './resnet50.pth'
    torch.save(resnet50.state_dict(), PATH1)
    
    # 2. Fine-tuning with Masking

    # Load the trained model weights into a new ResNet50 model for fine-tuning
    resnet50_for_finetuning = models.resnet50(pretrained=False)
    resnet50_for_finetuning.fc = nn.Linear(resnet50.fc.in_features, num_classes)
    resnet50_for_finetuning.load_state_dict(torch.load(PATH1))
    resnet50_for_finetuning.to(device)
    
    # Initialize the AttentionMaskingResNet50 model with the trained model weights
    attention_resnet50 = AttentionMaskingResNet50(resnet50_for_finetuning, num_classes).to(device)
    
    # Prepare the model for fine-tuning
    for param in attention_resnet50.parameters():
        param.requires_grad = True  # makes parameters modifiable for fine-tuning

    # Initialize the optimizer for the new model with a lower learning rate
    optimizer_ft = optim.SGD(attention_resnet50.parameters(), lr=0.0001, momentum=0.9)

    # Use only 1 epoch for fine-tuning with masking
    fine_tune_epochs = 1

    # Fine-tuning the model
    train_model(attention_resnet50, train_loader, criterion, optimizer_ft, fine_tune_epochs, device)
    
    # Save the fine-tuned model
    PATH2 = './attention_resnet50.pth'
    torch.save(attention_resnet50.state_dict(), PATH2)
