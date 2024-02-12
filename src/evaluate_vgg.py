import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import torch.optim as optim
from vgg import VGGWithAttention

def evaluate_model(model, dataloader, classes):
    model.eval()  # Set the model to evaluation mode
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, _ = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuratezza per la classe {classname}: {accuracy:.2f}%')

    # Calculate and print the overall accuracy
    total_correct = sum(correct_pred.values())
    total = sum(total_pred.values())
    overall_accuracy = 100 * float(total_correct) / total
    print(f'\nAccuratezza totale della rete sul test set: {overall_accuracy:.2f}%')



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

    # Initialize the model
    model = VGGWithAttention()

    # Define the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Load the trained model state
    PATH1 = './vgg_initial.pth'  
    model.load_state_dict(torch.load(PATH1, map_location=device))
    
    # After training, evaluate the model
    print('\nValutazione del modello sul test set senza masking:')
    evaluate_model(model, testloader, classes)
    
    # Load the state of the model after fine-tuning
    PATH2 = './vgg_finetuned.pth' 
    model.load_state_dict(torch.load(PATH2, map_location=device))
    
    # After training, evaluate the model
    print('\nValutazione del modello sul test set con masking:')
    evaluate_model(model, testloader, classes)