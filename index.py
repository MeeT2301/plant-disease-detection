import os
import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms, models  # datsets  , transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from flask import Flask, render_template,request
from PIL import Image
import torch
import torchvision.transforms.functional as TF
app = Flask(__name__,static_folder="static")
app.config['UPLOAD_FOLDER'] = os.path.join("static","upload_images")

class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)

        # Flatten
        out = out.view(-1, 50176)

        # Fully connected
        out = self.dense_layers(out)

        return out
def single_prediction(image_path,model):
    data = pd.read_csv("C:/Users/Admin/Downloads/Plant disease detector/Plant disease detector/disease_info.csv", encoding="cp1252")
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    pred_csv = data["disease_name"][index]
    plant_data = {}
    plant_data["disease"]=pred_csv
    plant_data["description"]=data["description"][index]
    plant_data["remedy"]=data["Possible Steps"][index]
    return plant_data

@app.route("/")
def index_page():
  return render_template('index.html')

@app.route("/upload",methods=["GET","POST"])
def upload_files():
    if request.method == "POST":
        #fetching user provided files
        if "usr_img" not in request.files:
            print(request.files)
            return "No File Provided"
        usr_img = request.files["usr_img"]
        #saving file to the upload folder
        path = os.path.join(app.config["UPLOAD_FOLDER"],usr_img.filename)
        usr_img.save(path)
        
        #loading model file
        targets_size = 39
        model = CNN(targets_size)
        model.load_state_dict(torch.load("C:/Users/Admin/Downloads/Plant disease detector/Plant disease detector/plant_disease_model_1.pt",map_location=torch.device('cpu')))
        model.eval()
        
        op = single_prediction(usr_img,model)
        return render_template("info.html",disease=op["disease"],description=op["description"],remedy=op["remedy"],link_img=path)
