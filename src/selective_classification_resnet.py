import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import torchvision.models as models
import torch.optim as optim
from resnet50 import test_dataset, test_loader, num_classes, AttentionMaskingResNet50
from sklearn.metrics import precision_recall_fscore_support

def selective_classification(model_initial, model_attention, dataloader, confidence_threshold):
    model_initial.eval()  # Imposta il primo modello in modalità valutazione
    model_attention.eval()  # Imposta il secondo modello in modalità valutazione
    total_samples = 0
    total_correct = 0
    rejected = 0
    agreements = 0  # Contatore per i casi in cui i modelli sono d'accordo

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Ottiene le predizioni da entrambi i modelli
            outputs_initial = model_initial(images)
            outputs_attention = model_attention(images)
            
            # Applica sigmoid per ottenere le probabilità per entrambi i modelli
            probs_initial = torch.sigmoid(outputs_initial)
            probs_attention = torch.sigmoid(outputs_attention)
            
            # Calcola le confidenze massime delle previsioni del modello di attenzione
            confidences_attention, _ = probs_attention.max(1)
            
            # Verifica se le confidenze superano la soglia
            confident_preds = confidences_attention > confidence_threshold
            
            # Determina le predizioni binarie per entrambi i modelli
            preds_initial = (probs_initial > 0.5).float()
            preds_attention = (probs_attention > 0.5).float()
            
            # Verifica l'accordo tra i modelli per ogni etichetta
            agreement = (preds_initial == preds_attention) & confident_preds.unsqueeze(1)
            mask_agreement = agreement.all(dim=1)  # Verifica l'accordo su tutte le etichette
            
            selected_labels = labels[mask_agreement]
            selected_preds = preds_attention[mask_agreement]
            
            if len(selected_labels) == 0:  # Se non ci sono predizioni concordate, continua al prossimo batch
                rejected += len(images)
                continue
            
            agreements += mask_agreement.sum().item()  # Aggiorna il conteggio dei casi di accordo
            correct_preds = (selected_preds == selected_labels).float().sum(dim=1)
            
            total_samples += len(selected_labels)
            total_correct += correct_preds.sum().item()

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    rejection_rate = (rejected / len(dataloader.dataset)) * 100
    agreement_rate = (agreements / len(dataloader.dataset)) * 100  # Calcola il tasso di accordo
    
    print(f'Accuracy (excluding rejections): {accuracy:.2f}%')
    print(f'Rejection Rate: {rejection_rate:.2f}%')
    print(f'Agreement Rate: {agreement_rate:.2f}%')  # Stampa il tasso di accordo




if __name__ == '__main__':
    
    # Definisci il dispositivo
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Carica il modello ResNet50 addestrato senza componenti di masking
    resnet50 = models.resnet50(pretrained=False)  # Inizializza il modello ResNet50
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)  # Aggiusta l'ultimo layer per num_classes
    PATH_resnet50 = './resnet50.pth'
    resnet50.load_state_dict(torch.load(PATH_resnet50, map_location=device))
    resnet50.to(device)
    
    # Carica il modello AttentionMaskingResNet50 addestrato
    attention_resnet50 = AttentionMaskingResNet50(resnet50, num_classes)  # Utilizza il modello resnet50 come base
    PATH_attention_resnet50 = './attention_resnet50.pth'
    attention_resnet50.load_state_dict(torch.load(PATH_attention_resnet50, map_location=device))
    attention_resnet50.to(device)

     # Test del modello con selective classification
    print("\nTesting con selective classification:")
    confidence_threshold = 0.6  # Soglia di confidenza per la selective classification
    selective_classification(resnet50, attention_resnet50, test_loader, confidence_threshold)