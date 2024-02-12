import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import torch.optim as optim
from vgg import VGGWithAttention

def selective_classification(vgg_initial, vgg_finetuned, dataloader, base_confidence_threshold):
    vgg_initial.eval()  # Set the first model to evaluation mode
    vgg_finetuned.eval()  # Set the second model to evaluation mode
    total, correct, rejected, agreements = 0, 0, 0, 0

    # Define weights for more difficult classes (bird, cat, deer, dog)
    class_difficulty_weights = {3: 0.7, 4: 0.7, 5: 0.7, 6: 0.7}
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            
            # Select only the model's output if it returns a tuple
            outputs_initial_tuple = vgg_initial(images)
            outputs_initial = outputs_initial_tuple[0] if isinstance(outputs_initial_tuple, tuple) else outputs_initial_tuple
            softmax_scores_initial = F.softmax(outputs_initial, dim=1)
            _, predictions_initial = torch.max(softmax_scores_initial, 1)
            
            outputs_finetuned_tuple = vgg_finetuned(images)
            outputs_finetuned = outputs_finetuned_tuple[0] if isinstance(outputs_finetuned_tuple, tuple) else outputs_finetuned_tuple
            softmax_scores_finetuned = F.softmax(outputs_finetuned, dim=1)
            confidences_finetuned, predictions_finetuned = torch.max(softmax_scores_finetuned, 1)
            
            for idx, (confidence_finetuned, prediction_finetuned) in enumerate(zip(confidences_finetuned, predictions_finetuned)):
                # Apply a weight to the confidence threshold if the class is considered difficult
                adjusted_confidence_threshold = base_confidence_threshold * class_difficulty_weights.get(prediction_finetuned.item(), 1)
                
                if prediction_finetuned == predictions_initial[idx]:
                    agreements += 1
                    if confidence_finetuned.item() >= adjusted_confidence_threshold:
                        total += 1
                        correct += (prediction_finetuned == labels[idx]).item()
                    else:
                        rejected += 1

    accuracy = correct / total if total > 0 else 0
    rejection_rate = rejected / (total + rejected)
    agreement_rate = agreements / len(dataloader.dataset)
    
    print(f'Accuracy (excluding rejections): {accuracy*100:.2f}%')
    print(f'Rejection Rate: {rejection_rate*100:.2f}%')
    print(f'Agreement Rate: {agreement_rate*100:.2f}%')


if __name__ == '__main__':
    
    # Define data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Initialize models
    model_initial = VGGWithAttention()
    model_finetuned = VGGWithAttention()

    # Define the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_initial.to(device)
    model_finetuned.to(device)

    # Load the trained model states
    PATH_INITIAL = './vgg_initial.pth'  
    PATH_FINETUNED = './vgg_finetuned.pth'  
    model_initial.load_state_dict(torch.load(PATH_INITIAL, map_location=device))
    model_finetuned.load_state_dict(torch.load(PATH_FINETUNED, map_location=device))
    

    # Test the model with selective classification
    print("\nTesting con selective classification:")
    confidence_threshold = 0.8  # Confidence threshold for selective classification
    selective_classification(model_initial, model_finetuned, testloader, confidence_threshold)