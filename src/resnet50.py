#Preparazione del dataset CelebA
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


# Percorsi per le immagini e per gli attributi
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



# Definisci le trasformazioni (ad esempio ridimensionamento e normalizzazione)
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

# Definisci la dimensione del subset
subset_size = 1000  # Ad esempio, usa 1000 immagini per l'addestramento

# Genera indici casuali per il subset
train_indices = torch.randperm(len(train_dataset))[:subset_size]

# Crea il subset del dataset di addestramento
train_subset = Subset(train_dataset, train_indices)

# Crea il DataLoader per il subset del dataset di addestramento
train_loader_subset = DataLoader(train_subset, batch_size=32, shuffle=True)


# Definizione del modello AttentionMaskingResNet50
class AttentionMaskingResNet50(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(AttentionMaskingResNet50, self).__init__()
        # Utilizza le caratteristiche del modello pre-addestrato, escludendo gli ultimi layer
        self.features = nn.Sequential(*list(pretrained_model.children())[:-2])
        self.avgpool = pretrained_model.avgpool
        # Aggiunge un generatore di maschere basato sull'output del penultimo layer
        self.mask_generator = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1),  # Assumendo che l'output delle caratteristiche abbia 2048 canali
            nn.Sigmoid()
        )
        # Utilizza il numero di caratteristiche in uscita dall'avgpool per il layer fully connected
        self.fc_in_features = pretrained_model.fc.in_features
        self.classifier = nn.Linear(self.fc_in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        mask = self.mask_generator(x)
        # Applica la maschera direttamente alle caratteristiche prima del pooling
        x = x * mask
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def calculate_metrics(outputs, labels):
    # Applica sigmoid per ottenere le probabilità
    probs = torch.sigmoid(outputs)
    # Converte le probabilità in previsioni binarie
    preds = (probs > 0.5).float()

    # True Positives, False Positives, True Negatives, False Negatives
    TP = (preds * labels).sum(dim=0)
    FP = ((1 - labels) * preds).sum(dim=0)
    FN = (labels * (1 - preds)).sum(dim=0)
    TN = ((1 - labels) * (1 - preds)).sum(dim=0)

    # Precisione, Recall, e F1 per ogni classe, poi calcola la media
    precision = (TP / (TP + FP + 1e-8)).mean()
    recall = (TP / (TP + FN + 1e-8)).mean()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1.item()




def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    print('Addestramento iniziato...')
    
    for epoch in range(num_epochs):  # loop sul dataset per un numero fissato di epoche
        running_loss = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for inputs, labels in train_loader: 
            # Ottiene gli input; i dati sono una lista di [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Azzera i gradienti dei parametri del modello
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()
            
            # Calcola le metriche per il batch corrente
            precision, recall, f1 = calculate_metrics(outputs, labels)
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Stampa statistiche di addestramento
            running_loss += loss.item()
            
        # Calcola la media delle metriche per l'epoca
        avg_precision = total_precision / len(train_loader)
        avg_recall = total_recall / len(train_loader)
        avg_f1 = total_f1 / len(train_loader)
        
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}')

    print('Addestramento completato')

    

if __name__ == '__main__':
    
    # Impostazione degli iperparametri
    learning_rate = 0.001
    momentum = 0.9
    batch_size = 32
    num_epochs = 20
    
    # Imposta il device su 'mps' per utilizzare Metal Performance Shaders su Mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Addestramento iniziale senza Masking
    # Carica una ResNet50 pre-addestrata senza componenti di masking
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
    resnet50.to(device)
    
    
    # Definizione della funzione di perdita per classificazione multi-etichetta
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(resnet50.parameters(), lr=learning_rate, momentum=momentum)
            
    # Chiamata alla funzione di addestramento sull'intero dataset
    train_model(resnet50, train_loader, criterion, optimizer, num_epochs, device)
    
    PATH1 = './resnet50.pth'
    torch.save(resnet50.state_dict(), PATH1)
    
    # 2. Fine-tuning con Masking

    # Carica i pesi del modello addestrato in un nuovo modello ResNet50 per il fine-tuning
    resnet50_for_finetuning = models.resnet50(pretrained=False)
    resnet50_for_finetuning.fc = nn.Linear(resnet50.fc.in_features, num_classes)
    resnet50_for_finetuning.load_state_dict(torch.load(PATH1))
    resnet50_for_finetuning.to(device)
    
    # Inizializzazione del modello AttentionMaskingResNet50 con i pesi del modello addestrato
    attention_resnet50 = AttentionMaskingResNet50(resnet50_for_finetuning, num_classes).to(device)
    
    # Prepara il modello per il fine-tuning
    for param in attention_resnet50.parameters():
        param.requires_grad = True  # Opzionale: rende i parametri modificabili per il fine-tuning

    # Inizializza l'ottimizzatore per il nuovo modello con learning rate più basso
    optimizer_ft = optim.SGD(attention_resnet50.parameters(), lr=0.0001, momentum=0.9)

    # Utilizza solo 1 epoca per il fine-tuning con masking
    fine_tune_epochs = 1

    # Fine-tuning del modello
    train_model(attention_resnet50, train_loader, criterion, optimizer_ft, fine_tune_epochs, device)
    
    # Salva il modello fine-tunato
    PATH2 = './attention_resnet50.pth'
    torch.save(attention_resnet50.state_dict(), PATH2)
