import methods as m
from preparaData import prepare

import time
import csv
import os

from torchvision import models
from torchinfo import summary
import torch
from ptflops import get_model_complexity_info


IMG_DIR = "./imagenet-sample-images"
SAMPLE_SIZE = 100


#rodzaje pruningu
modes = list(range(6))

#flops modelu w MegaFlopach
def measureFlops(model):
    inputRes = (3,224,224)
    flops, _ = get_model_complexity_info(model, inputRes, as_strings=False, print_per_layer_stat=False)
    flops *= 2
    flops /= 1000000
    return flops

#sprawdza rozmiar modelu po pruningu
def checkModelSize(model, device):
    info = summary(
        model,
        input_size=(1,3,224,224),
        device=device,
        verbose=0
    )


    input_MB = toBytes(info.input_size) / 1024**2
    forward_MB = toBytes(info.total_output_bytes) / 1024**2
    params_MB = toBytes(info.total_param_bytes) / 1024**2
    total_MB = input_MB + forward_MB + params_MB

    memory_data = {
        "Metric": [
            "Input size (MB)",
            "Forward pass size (MB)",
            "Params size (MB)",
            "Estimated Total Size (MB)"
        ],
        "Value_MB": [
            input_MB,
            forward_MB,
            params_MB,
            total_MB
        ]
    }

    return total_MB
def toBytes(x, dtype_bytes=4):
    """
    Zamienia torchinfo input_size / output_size / param_bytes na bajty.
    Obsługuje tuple, listy i zagnieżdżenia.
    dtype_bytes = 4 dla float32
    """
    # jeśli to int, zwróć bez zmian
    if isinstance(x, int):
        return x
    # jeśli to float, zwróć jako int (rzadko)
    if isinstance(x, float):
        return int(x)
    
    # jeśli lista / tuple
    if isinstance(x, (list, tuple)):
        total = 0
        for elem in x:
            # jeśli element jest liczbą, pomnóż przez dtype_bytes
            if isinstance(elem, int):
                total += elem * dtype_bytes
            # jeśli element jest kolejnym tuple/listą -> rekurencja
            elif isinstance(elem, (list, tuple)):
                total += toBytes(elem, dtype_bytes=dtype_bytes)
            else:
                raise TypeError(f"Nieobsługiwany typ: {type(elem)} w {elem}")
        return total
    
    raise TypeError(f"Nieobsługiwany typ: {type(x)}")


#mierzy poprawnosc rozpoznawania obrazow modelu, czas i zuzycie pamieci
def measureInference(model, imgDir, sampleSize, device):
    images, labels = prepare(imgDir, sampleSize)

    images = images.to(device)
    labels = labels.to(device)

    elapsed = 0
    correct = 0
    correctTop5 = 0 #metryka top5 accuracy, czy prawdziwa klasa jest w top 5 predykcjach

        
    for i in range(images.shape[0]):

        with torch.no_grad():
            img = images[i].unsqueeze(0)
            label = labels[i]

            start = time.perf_counter()
            output = model(img)
            end = time.perf_counter()

            _, pred = torch.max(output, 1)

            _, top5 = output.topk(5,dim=1)
            correctTop5 += int(label.item() in top5[0])

            if pred.item() == label.item():
                correct+=1

            elapsed += end-start
    

    avgTime = elapsed/sampleSize*1000   #sredni czas pomiaru w ms
    accuracy = correct/sampleSize*100   #dokladnosc modelu
    top5Accuracy = correctTop5/sampleSize*100
    return[accuracy, top5Accuracy, avgTime]

#eksport do csv
def exportToCSV(name, cutoff, size, results, flops):
    folder = os.path.join("..", "results")

    os.makedirs(folder, exist_ok=True)

    fileName = name + ".csv"

    file = os.path.join(folder, fileName)
    
    with open(file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["poziom uciecia", "rozmiar [MB]", "dokladnosc [%]", "dokladnosc top5 [%]", "sredni czas inferencji [ms]", "moc obliczeniowa [MFLOPS]"])

        for i in range(len(size)):
            writer.writerow([cutoff[i], size[i], results[i][0], results[i][1], results[i][2], flops[i]])


def main():
    cutoff = [1, 2, 5, 10, 20, 50, 100]
    for num in modes:       #dla kazdej wersji pruningu
        print(f"test nr {num}")
        avgSize = []
        avgResults = []
        avgFlops = []
        for limit in cutoff:     #dla stopnia obciecia
            print(f"iteracja nr {limit}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT).eval()

            amount = limit/100
            size = []
            results = []
            flops = []
            model, name = m.getPruning(num, model, amount)
            if num != 5:
                model = m.removeMasks(model)

            #sparsity = m.checkSparsity(model)

            for j in range(10):  #10 powtorzen pomiarow
                #pruning:
                size.append(checkModelSize(model, device))
                results.append (measureInference(model, IMG_DIR, SAMPLE_SIZE, device))
                flops.append(measureFlops(model))
            

            avgSize.append(sum(size)/len(size))
            avgResults.append([sum(results[0])/len(results[0]),sum(results[1])/len(results[1]),sum(results[2])/len(results[2])])
            avgFlops.append(sum(flops)/len(flops))
        exportToCSV(name, cutoff, avgSize, avgResults, avgFlops)
            

if __name__ == '__main__':
    main()


            