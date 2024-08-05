import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Parámetros
img_size = 224  # Tamaño de imagen recomendado para ResNet-34
batch_size = 100
num_epochs = 20  # Ajusta el número de épocas si es necesario
n_splits = 2  # Número de splits para KFold

# Transformaciones de datos
transform = transforms.Compose([
    transforms.Resize(256),  # Aumentar el tamaño de la imagen
    transforms.CenterCrop(img_size),  # Recortar al tamaño deseado
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar datos
dataset = datasets.ImageFolder(root='Entrenamiento', transform=transform)

# Definir el modelo
def create_model(num_classes):
    model = models.resnet34(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Mover el modelo a la GPU si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurar el optimizador y la función de pérdida
criterion = nn.CrossEntropyLoss()

# Función de entrenamiento
def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_batches = len(data_loader)
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        # Imprimir el progreso del lote
        if (batch_idx + 1) % 10 == 0:  # Mostrar cada 10 lotes
            print(f'Epoch Progress - Batch {batch_idx + 1}/{total_batches}...')
    
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss

# Función de evaluación
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    val_loss = running_loss / len(data_loader.dataset)
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, val_loss, cm

# Crear las carpetas necesarias si no existen
if not os.path.exists('graficas'):
    os.makedirs('graficas')
if not os.path.exists('model'):
    os.makedirs('model')

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
all_loss = []
all_val_loss = []
all_confusion_matrices = []

for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}/{n_splits}')
    
    train_subsampler = SubsetRandomSampler(train_index)
    val_subsampler = SubsetRandomSampler(val_index)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
    
    model = create_model(num_classes=len(dataset.classes))
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    fold_loss = []
    fold_val_loss = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        fold_loss.append(train_loss)
        print(f'Train Loss: {train_loss:.4f}')
        
        val_accuracy, val_loss, cm = evaluate_model(model, val_loader, criterion, device)
        fold_val_loss.append(val_loss)
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        
    all_loss.append(fold_loss)
    all_val_loss.append(fold_val_loss)
    all_confusion_matrices.append(cm)

# Guardar el modelo final después de todos los pliegues
torch.save(model.state_dict(), 'model/modelo_final.pth')

# Sumar las matrices de confusión
mean_confusion_mtx = np.sum(all_confusion_matrices, axis=0)

# Crear un colormap cian
cian_cmap = sns.color_palette("Blues", as_cmap=True)

# Guardar la matriz de confusión sumada
plt.figure(figsize=(16, 16))
sns.heatmap(mean_confusion_mtx.astype(int), annot=True, fmt='d', cmap=cian_cmap, xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.savefig('graficas/confusion_matrix.png')
plt.show()

# Promedio de pérdidas a través de todos los pliegues
mean_loss = np.mean([np.mean(fold) for fold in all_loss])
mean_val_loss = np.mean([np.mean(fold) for fold in all_val_loss])

# Crear la gráfica del historial de pérdida promedio
plt.figure()
for fold_loss in all_loss:
    plt.plot(fold_loss, label='Train Loss')
for fold_val_loss in all_val_loss:
    plt.plot(fold_val_loss, label='Validation Loss', linestyle='--')
plt.legend()
plt.title("Historial de pérdida promedio")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('graficas/loss.png')
plt.show()
