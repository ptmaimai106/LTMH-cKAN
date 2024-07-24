import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader

from architectures_28x28.CKAN_BN import CKAN_BN
from architectures_28x28.SimpleModels import *
from architectures_28x28.ConvNet import ConvNet
from architectures_28x28.KANConvs_MLP import KANC_MLP
from architectures_28x28.KKAN import KKAN_Convolutional_Network
from architectures_28x28.conv_and_kan import NormalConvsKAN

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Cargar MNIST y filtrar por dos clases
mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)


def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
    return test_loss / len(test_loader)

def train_and_save_best_value(model, device, train_loader, test_loader, optimizer, scheduler, criterion, save_dir, num_epochs):
    """
    Train the model and save the model with the best validation loss

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation data
        optimizer: the optimizer to use (e.g. AdamW)
        scheduler: the learning rate scheduler
        criterion: the loss function (e.g. CrossEntropy)
        save_dir: the directory to save the model
        num_epochs: number of epochs to train

    Returns:
        best_test_loss: the best validation loss achieved
    """

    model.to(device)
    train_and_save_best_value.best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = evaluate(model, device, test_loader, criterion)
        
        # Adjust the learning rate
        scheduler.step()

        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the model if it has the best validation loss
        if avg_test_loss < train_and_save_best_value.best_test_loss:
            train_and_save_best_value.best_test_loss = avg_test_loss
            model_save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Epoch {epoch+1}: Model saved to {model_save_path} with validation loss {avg_test_loss:.6f}')
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_test_loss:.6f}')
    
    return train_and_save_best_value.best_test_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("CUDA is available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

model_KKAN_Convolutional_Network = KKAN_Convolutional_Network(device=device)
model_KKAN_Convolutional_Network.to(device)
optimizer_KKAN_Convolutional_Network = optim.AdamW(model_KKAN_Convolutional_Network.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_KKAN_Convolutional_Network = optim.lr_scheduler.ExponentialLR(optimizer_KKAN_Convolutional_Network, gamma=0.8)
criterion_KKAN_Convolutional_Network = nn.CrossEntropyLoss()

num_epochs = 10

save_dir = 'models/cKAN_model_10_epoch'
best_loss = train_and_save_best_value(
    model=model_KKAN_Convolutional_Network, 
    device=device, 
    train_loader=train_loader, 
    test_loader=test_loader,   
    optimizer=optimizer_KKAN_Convolutional_Network, 
    scheduler=scheduler_KKAN_Convolutional_Network, 
    criterion=criterion_KKAN_Convolutional_Network, 
    save_dir=save_dir, 
    num_epochs=num_epochs
)

print(f'Training complete. Best validation loss: {best_loss:.6f}')

