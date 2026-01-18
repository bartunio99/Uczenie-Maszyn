from torchvision import datasets
import numpy as np
import os
import random

# ------------------------- KONFIG -------------------------
OUTPUT_FILE = "cifar10_32x32.h"  # plik wyjściowy .h
NUM_IMAGES = 20                   # liczba losowych obrazów
IMAGE_SIZE = 32                   # CIFAR-10 ma 32x32
# -----------------------------------------------------------

def importCIFAR():
    # Pobranie datasetu CIFAR-10
    cifar10 = datasets.CIFAR10(root="./data", train=False, download=True)

    # Wybór losowych indeksów
    indices = random.sample(range(len(cifar10)), NUM_IMAGES)

    images_int8 = []
    labels = []

    for idx in indices:
        img, label = cifar10[idx]
        labels.append(label)

        # Przetwarzanie obrazu
        img_np = np.array(img).astype(np.float32) / 255.0        # normalizacja 0-1
        img_int8 = (img_np * 255 / 2 - 128).astype(np.int8)      # INT8 [-128,127]
        img_int8 = np.transpose(img_int8, (2, 0, 1))             # CHW
        images_int8.append(img_int8)

    images_int8 = np.array(images_int8)
    labels = np.array(labels, dtype=np.uint16)

    # Zapis do pliku .h
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"#define NUM_TEST_IMAGES {NUM_IMAGES}\n")
        f.write(f"#define IMAGE_HEIGHT {IMAGE_SIZE}\n")
        f.write(f"#define IMAGE_WIDTH {IMAGE_SIZE}\n")
        f.write(f"#define IMAGE_CHANNELS 3\n\n")

        f.write(f"const uint16_t test_labels[{NUM_IMAGES}] = {{")
        f.write(",".join(str(l) for l in labels))
        f.write("};\n\n")

        f.write(f"const int8_t test_images[{NUM_IMAGES}][IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_CHANNELS] = {{\n")
        for n in range(NUM_IMAGES):
            f.write("  {")
            f.write(",".join(str(int(v)) for v in images_int8[n].flatten()))
            f.write("},\n")
        f.write("};\n")

    print(f"Plik '{OUTPUT_FILE}' wygenerowany, gotowy do ESP32.")

if __name__ == "__main__":
    importCIFAR()
