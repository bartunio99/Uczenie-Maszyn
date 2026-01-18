import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import tensorflow as tf
import onnx
import onnx2tf.onnx2tf as o2t
from torchvision import models
import os



weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
sample_inputs = (torch.randn(1, 3, 224, 224),)
torch_output = model(*sample_inputs)

def main():
    pruning(model)
    checkSparsity(model)
    removeMasks(model)
    torch.save(model, "models/pt/resnet18_pruned.pt")

    exportModel(model)

#przyciecie dla kazdej warstwy
def pruning(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):#filtrowanie konwolucji - zajmuja znaczna wiekszosc rozmiaru calego modelu, reszta insignificant
            prune.random_unstructured(
                module,
                name="weight",
                amount=0.3
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
        
    #konwersja do ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "models/onnx/resnet18.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=12,  
        do_constant_folding=True
    )
    print("ONNX zapisany jako resnet.onnx")


    #konwersja do tf
    saved_model_dir = "models/tf/resnet18_tf"
    os.makedirs(saved_model_dir, exist_ok=True)

    # Wczytanie ONNX jako ModelProto
    onnx_model = onnx.load("models/onnx/resnet18.onnx")

    # Konwersja do TensorFlow
    o2t.convert(onnx_model, saved_model_dir)
    print(f"TensorFlow SavedModel zapisany w {saved_model_dir}")
    print("1")
    #konwersja do tflite micro
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    print("2")
    # Opcjonalna kwantyzacja int8 dla mikrokontrolerów
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("3")
    def representative_dataset():
        for i in range(10):
            print(f"Processing representative sample {i+1}/10")
            yield [tf.random.normal([1, 224, 224, 3], dtype=tf.float32)]
            

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open("onnx/tflite/resnet.tflite", "wb") as f:
        f.write(tflite_model)

    print("TFLite Micro zapisany jako mobilenet_v2_micro.tflite")




if __name__ == "__main__":
    main()
 
