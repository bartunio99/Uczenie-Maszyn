#Model rozpoznajacy czlowieka
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

#Inferencja obrazu
def evaluate(img):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()


    with torch.no_grad():
        output = model(img)

    #najblizsza klasa do outputu
    predicted_class = output.argmax(dim=1)

    #konwersja liczby na nazwe klasy
    labels = weights.meta["categories"]
    id = predicted_class.item()

    return(labels[id], id)


#normalizacja obrazu
def normalize(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(image)
    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def main():
    img = "images/pan.jpg"
    rtn = normalize(img)
    cls = evaluate(rtn)
    print(cls)

if __name__ == "__main__":
    main()