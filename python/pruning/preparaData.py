#przygotowanie danych do pomiarow
import torch
from torchvision import transforms, datasets, models
from PIL import Image
import os
import random
from torchvision.models import squeezenet1_1
import subprocess


REPO_PATH = "https://github.com/EliSchwartz/imagenet-sample-images.git"

model = squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
class_names = model.classes if hasattr(model, 'classes') else None 

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229,0.224,0.225])
])


def prepare(dir, sampleSize):
    if os.path.isdir(dir) == False:
        subprocess.run(["git", "clone" ,REPO_PATH])

    images = []
    labels = []
    files = [f for f in os.listdir(dir) if f.endswith(".JPEG")]

    for i in range(sampleSize):
        index = random.randint(0,999)
        file = files[index]
        path = os.path.join(dir, file)
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)
        labels.append(index)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels



