from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import torch
from torchvision.models import resnet18
from torchvision import transforms
import torch.nn as nn
import os
from PIL import Image

app = Flask(__name__)

# Modelcle
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        # Load pre-trained ResNet model
        self.pre_model = resnet18(pretrained=True)
        # Freeze the pre-trained layers
        for param in self.pre_model.parameters():
            param.requires_grad = False

        # Replace the fully connected layer
        self.pre_model.fc = nn.Linear(self.pre_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.pre_model(x)

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Ensure the image has 3 channels
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Define the number of classes
num_classes = 2
classes = ('Normal', 'Pneumonia')

# Load model
model_path = r'C:\Users\nisha\Desktop\ALJO\DEEP LEARNING\Pneumonia\Pneumonia\chest_xray.pth'
model = ResNet18(num_classes)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Ensure the directory for saving uploaded images exists
UPLOADS_DIR = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        # Uploaded image
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        try:
            img = Image.open(file)
        except Exception as e:
            return render_template('index.html', error=str(e))
        
        img_tensor = transform(img).unsqueeze(0)

        # Handle model prediction
        with torch.no_grad():
            output = model(img_tensor)
        
        # Get predicted class label
        _, predicted_index = torch.max(output, 1)
        predicted_label = classes[predicted_index.item()]

        # Save uploaded images with prediction label
        img_filename = secure_filename(file.filename)
        img_path = os.path.join(UPLOADS_DIR, img_filename)
        img.save(img_path)

        # Generate URL for the uploaded image
        img_url = url_for('static', filename=f'uploads/{img_filename}')

        return render_template('index.html', img_url=img_url, predicted_label=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)
