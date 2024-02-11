# Import delle librerie necessarie

import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


# Costruzione del Modello Base (Una versione ridotta di VGG)
# Definire un Nuovo Modello con Accesso a uno Strato Intermedio:
# Prima di tutto, definiremo una versione modificata del modello che ci permetta di accedere alle attivazioni di uno strato intermedio, che useremo come nostra "mappa di attenzione".

class VGGWithAttention(nn.Module):
    def __init__(self):
        super(VGGWithAttention, self).__init__()
        # Strato convoluzionale 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Input: 3 canali, Output: 32 canali, Kernel: 3x3
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)  # Kernel: 2x2
        # Strato convoluzionale 2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        # Strato convoluzionale 3 (ridotto rispetto alla VGG standard)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        # Strati fully connected ridotti
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Ridotto da 4096
        self.fc2 = nn.Linear(512, 256)  # Ridotto da 4096 a 256
        self.fc3 = nn.Linear(256, 10)  # 10 classi di output per CIFAR-10

    def forward(self, x):
        # Applicazione degli strati convoluzionali e pooling
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        # Considera le attivazioni qui come mappa di attenzione
        attention_map = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.pool(attention_map)
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output, attention_map


# Funzione per l'implementazione del Masking

def apply_attention_masking(inputs, model, attenuation_factor=0.5, dynamic_threshold=True):
    _, attention_maps = model(inputs)

    if dynamic_threshold:
        threshold = attention_maps.quantile(0.75)
    else:
        threshold = attention_maps.mean()

    attention_maps_normalized = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min())

    # Fai l'upsampling delle mappe di attenzione normalizzate per corrispondere alle dimensioni dell'input
    size = (inputs.size(2), inputs.size(3))  # Dimensioni spaziali dell'input originale
    attention_maps_upsampled = F.interpolate(attention_maps_normalized, size=size, mode='bilinear', align_corners=False)

    # Calcola la maschera usando il fattore di attenuazione
    masks = (attention_maps_upsampled > threshold).float() * attenuation_factor

    # Riduci le dimensioni dei canali della maschera a 1 per il broadcasting corretto
    masks = masks.mean(dim=1, keepdim=True)

    # Applica la maschera attenuata agli input
    masked_inputs = inputs * (1 - masks)

    return masked_inputs


def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = (correct / total) * 100
    return accuracy

def train_model(model, trainloader, criterion, optimizer, num_epochs, device, apply_mask):
    for epoch in range(num_epochs):  # loop sul dataset per un numero fissato di epoche
        running_loss = 0.0
        total_accuracy = 0.0

        for i, data in enumerate(trainloader, 0):
            # Ottiene gli input; i dati sono una lista di [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Applica l'attention masking se richiesto
            if apply_mask:
                inputs = apply_attention_masking(inputs, model)

            # Azzera i gradienti dei parametri del modello
            optimizer.zero_grad()

            # Forward pass con gli input mascherati
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()

            # Stampa statistiche di addestramento
            running_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)
            if i % 2000 == 1999:  # stampa ogni 2000 mini-batch
                print('[%d, %5d] loss: %.3f accuracy: %.2f%%' %
                      (epoch + 1, i + 1, running_loss / 2000, total_accuracy / 2000))
                running_loss = 0.0
                total_accuracy = 0.0



if __name__ == '__main__':
    
    # Definizione delle trasformazioni per il preprocessing del dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convertire le immagini in Tensori PyTorch
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizzazione
    ])

    # Caricamento del dataset CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Classi CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("Dataset CIFAR-10 pronto per l'uso.")

    # Impostazione degli iperparametri
    learning_rate = 0.001
    momentum = 0.9
    batch_size = 4
    num_epochs = 12

    # Imposta il device su 'mps' per utilizzare Metal Performance Shaders su Mac
    """MPS è un framework avanzato di Apple che fornisce un'ampia gamma di shader di elaborazione 
    delle immagini e di calcolo scientifico ottimizzati per sfruttare l'hardware GPU (Graphics Processing Unit) dei dispositivi Mac."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


    # Inizializzazione del modello e spostamento sul dispositivo appropriato
    model = VGGWithAttention().to(device)

    # Definizione della funzione di perdita e dell'ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Caricamento dei dati con le dimensioni del batch specificate
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)  
    
    # Chiamata alla funzione di addestramento
    # Addestramento iniziale senza masking
    print("Addestramento iniziale senza masking...")
    train_model(model, trainloader, criterion, optimizer, num_epochs=num_epochs, device=device, apply_mask=False)

    # Salvataggio del modello addestrato
    torch.save(model.state_dict(), './vgg_initial.pth')
    print("\nModello iniziale addestrato e salvato.")

    # Caricamento del modello addestrato
    model.load_state_dict(torch.load('./vgg_initial.pth'))
    print("\nModello iniziale caricato per il fine-tuning.")

    # Configurazione dell'ottimizzatore per il fine-tuning
    # È consigliabile utilizzare un tasso di apprendimento più basso per il fine-tuning
    optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # Fine-tuning con masking per una singola epoca
    print("\nInizio fine-tuning con masking...")
    train_model(model, trainloader, criterion, optimizer_ft, num_epochs=1, device=device, apply_mask=True)

    # Salvataggio del modello fine-tunato
    torch.save(model.state_dict(), './vgg_finetuned.pth')
    print("\nFine-tuning completato e modello salvato.")
    