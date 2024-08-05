import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import base64

app = Flask(__name__)

# Definir el modelo
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.model = models.resnet34(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

num_classes = 13
model = CustomResNet(num_classes)

# Cargar el estado del modelo guardado
model_state = torch.load('model/modelo_final.pth', map_location=torch.device('cpu'))
model.model.load_state_dict(model_state)
model.eval()

classes = ['AderezoDeMostaza', 'Atun', 'Chamoy', 'CremaDeCacahuate', 'JarabeDeMapple', 
           'LataDeVerduras', 'Mayonesa', 'Mostaza', 'PureDeTomate', 'SalsaBbq', 
           'SalsaDeSoya', 'SalsaMaggi', 'SalsaRanch']

history_texts = {
    "AderezoDeMostaza": {
        "text": ["Calorias: 758 kcal\nGrasas: 20 g"],
        "image_path": "FotosDeFront/AderesoDeMostaza.jpeg"
    },
    "Atun": {
        "text": ["Calorias: 107.65 kcal\nProteinas: 12.26 g"],
        "image_path": "FotosDeFront/Atun.jpeg"
    },
    "Chamoy": {
        "text": ["Calorias: 199.70 kcal\nAzucares: 3.95 g"],
        "image_path": "FotosDeFront/Chamoy.jpeg"
    },
    "CremaDeCacahuate": {
        "text": ["Calorias: 2975.60 kcal\nProteinas: 22.49 g"],
        "image_path": "FotosDeFront/CremaDeCacahuate.jpeg"
    },
    "JarabeDeMapple": {
        "text": ["Calorias: 1035 kcal\nAzucares: 50 g"],
        "image_path": "FotosDeFront/JarabeDeMapple.jpeg"
    },
    "LataDeVerduras": {
        "text": ["Calorias: 78 kcal\nProteinas: 2 g"],
        "image_path": "FotosDeFront/LataDeVerduras.jpeg"
    },
    "Mayonesa": {
        "text": ["Calorias: 3750 kcal\nGrasas: 82 g"],
        "image_path": "FotosDeFront/Mayonesa.jpeg"
    },
    "Mostaza": {
        "text": ["Calorias: 73.25 kcal\nGrasas: 4.25 g"],
        "image_path": "FotosDeFront/Mostaza.jpeg"
    },
    "PureDeTomate": {
        "text": ["Calorias: 50 kcal\nSodio: 320 mg"],
        "image_path": "FotosDeFront/PureDeTomate.jpeg"
    },
    "SalsaBbq": {
        "text": ["Calorias: 1010 kcal\nAzucares: 26 g"],
        "image_path": "FotosDeFront/SalsaBbq.jpeg"
    },
    "SalsaDeSoya": {
        "text": ["Proteinas: 0.37 kcal\nSodio: 1500 mg"],
        "image_path": "FotosDeFront/SalsaDeSoya.jpeg"
    },
    "SalsaMaggi": {
        "text": ["Proteinas: 5.27 g\nGrasas: 2 g"],
        "image_path": "FotosDeFront/SalsaMaggi.jpeg"
    }, 
    "SalsaRanch": {
        "text": ["Calorias: 340 kcal\nGrasas: 34.9 g"],
        "image_path": "FotosDeFront/SalsaRanch.jpeg"
    },
}

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.files['image']
        img = Image.open(image_data).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
        
        predicted_class = classes[predicted.item()]
        
        response_data = {'predicted_class': predicted_class}
        
        if predicted_class in history_texts:
            response_data['history_text'] = history_texts[predicted_class]['text']
            image_encoded = encode_image(history_texts[predicted_class]['image_path'])
            response_data['image'] = image_encoded
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
