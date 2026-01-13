import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import tensorflow as tf
import torch.optim as optim
from torchvision import models, transforms, datasets
from onnx_tf.backend import prepare
import onnx
import os
import numpy as np
from PIL import Image
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_DIR = "./imagenet-sample-images"
NUM_IMAGES_REPRESENTATIVE = 50        

model = models.squeezenet1_1(weights=None)

def squeezenet1_1_cifar(num_classes=10, model=model):
    

    model.conv1 = nn.Conv2d(
        3, 32, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(512, num_classes)

def train(model):
    model = model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=128)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch} done")

def main():
    squeezenet1_1_cifar(model=model)
    train(model)
    pruning_l1_struct(model)
    checkSparsity(model)
    removeMasks(model)
    torch.save(model, "models/pt/resnet18.pt")

    exportModel(model)

#przyciecie dla kazdej warstwy
def pruning_random_unstruct(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):#filtrowanie konwolucji - zajmuja znaczna wiekszosc rozmiaru calego modelu, reszta insignificant
            prune.random_unstructured(
                module,
                name="weight",
                amount=0.5
            )

def pruning_random_struct(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(
                module,
                name="weight",
                amount=0.5,
                dim=0
            )

# przy niestrukturalnym L1=L2        
def pruning_l1_unstruct(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(
                module,
                name="weight",
                amount=0.5
            )

def pruning_l1_struct(model):
     for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module,
                name="weight",
                amount=0.5,
                n=1,
                dim=0
            )

def pruning_l2_struct(model):
     for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module,
                name="weight",
                amount=0.5,
                n=2,
                dim=0
            )


#sprawdzenie gestosci modelu
def checkSparsity(model):
    zeros=0
    total=0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            zeros += torch.sum(module.weight == 0)
            total += module.weight.nelement()
    sparsity = 100*zeros/total
    print(f"Sparsity: {100 * zeros / total:.2f}%")

#usuwanie masek po pruningu
def removeMasks(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, "weight")

def exportModel(model):
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        "models/onnx/mobilenet_v2.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=12,  
        do_constant_folding=True
    )
    print("ONNX zapisany jako mobilenet_v2.onnx")

    saved_model_dir = "models/tf/mobilenet_v2_tf"
    os.makedirs(saved_model_dir, exist_ok=True)

    # Torch2TF konwersja
    # torch2tf.convert(model, dummy_input, output_dir)
    #konwersja do tf
    saved_model_dir = "models/tf/mobilenet_v2_tf"
    os.makedirs(saved_model_dir, exist_ok=True)

    # Wczytanie ONNX jako ModelProto
    onnx_model = onnx.load("models/onnx/mobilenet_v2.onnx")

    # Konwersja do TensorFlow
    tf_rep=prepare(onnx_model)
    tf_rep.export_graph(saved_model_dir)
    print(f"TensorFlow SavedModel zapisany w {saved_model_dir}")
    #konwersja do tflite micro
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Włączamy optymalizację
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Zbiór reprezentatywny (zwiększ do 100 próbek dla lepszej dokładności)
    transform = transforms.Compose([
    transforms.Resize((32, 32)),  # resize wejścia
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================
# Funkcja do representative dataset dla INT8
# ======================
    def get_representative_dataset():
        image_paths = glob.glob(os.path.join(IMG_DIR, "*.JPEG"))[:NUM_IMAGES_REPRESENTATIVE]
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = transform(img)
            img = img.unsqueeze(0).numpy().astype(np.float32)
            yield [img]

    converter.representative_dataset = get_representative_dataset

    # KLUCZOWE: Wymuszenie pełnej kwantyzacji stałoprzecinkowej
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # To wymusza, aby nawet operacje, które nie mają implementacji INT8, 
    # nie wracały do Float32 (wyrzuci błąd przy konwersji, jeśli czegoś brakuje)
    converter.target_spec.supported_types = [tf.int8] 

    # Ustawiamy wejście i wyjście na INT8
    converter.inference_input_type = tf.int8 
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open("models/tflite/resnet.tflite", "wb") as f:
        f.write(tflite_model)

    print("TFLite Micro zapisany jako mobilenet_v2_micro.tflite")

#eksport na plik esp



if __name__ == "__main__":
    main()
 
