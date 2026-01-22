#metody pruningu
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader, Dataset, IterableDataset
from preparaData import prepare
import torch.optim as optim
from datasets import load_dataset
from torchvision import transforms

import math

IMG_DIR = "./imagenet-sample-images"
SAMPLE_SIZE = 1000


#przyciecie dla kazdej warstwy
def pruning_random_unstruct(model, amount):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):#filtrowanie konwolucji - zajmuja znaczna wiekszosc rozmiaru calego modelu, reszta insignificant
            prune.random_unstructured(
                module,
                name="weight",
                amount=amount
            )
    return model

def pruning_random_struct(model, amount):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(
                module,
                name="weight",
                amount=amount,
                dim=0
            )
    return model

# przy niestrukturalnym L1=L2        
def pruning_l1_unstruct(model, amount):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(
                module,
                name="weight",
                amount=amount
            )
    return model

def pruning_l1_struct(model, amount):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=1,
                dim=0
            )
    return model


def pruning_l2_struct(model, amount):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=2,
                dim=0
            )
    return model


def pruning_l2_struct_conv10(model, amount):
    #pruning warstwy conv10
    conv10=model.classifier[1]
    W = conv10.weight.data
    out_channels=W.size(0)

    print(type(out_channels))


    prune_channels = int(math.floor(out_channels*amount))

    if prune_channels==0:
        return model
    
    l2_norms = torch.norm(
        W.view(out_channels, -1),
        p=2,
        dim=1
    )

    keep_idx = torch.argsort(l2_norms, descending=True)[
        : out_channels - prune_channels
    ]

    new_conv10 = nn.Conv2d(
        in_channels=conv10.in_channels,
        out_channels=len(keep_idx),
        kernel_size=conv10.kernel_size,
        stride=conv10.stride,
        padding=conv10.padding,
        dilation=conv10.dilation,
        groups=conv10.groups,
        bias=(conv10.bias is not None),
        padding_mode=conv10.padding_mode
    )

    new_conv10.weight.data = conv10.weight.data[keep_idx].clone()

    if conv10.bias is not None:
        new_conv10.bias.data = conv10.bias.data[keep_idx].clone()

    model.classifier[1] = new_conv10

    return model, "l2"


def prune_l1_struct_hard(model, amount, device):
    selected_filters = [12,13]  #fire8 i 9
    prev_keep = None

    
    prev_keep = None
    for idx, module in enumerate(model.features):
        if idx in selected_filters:
            model.features[idx], prev_keep = prune_fire_module(module, amount, prev_keep)
    in_channels = prev_keep.shape[0]  # prev_keep z ostatniego Fire module

    # zbuduj classifier ponownie
    num_classes = 1000  # lub Twój własny dataset
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(in_channels, num_classes, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1,1))
    )
    model = model.to(device)
    # model = fine_tune(model,device)


    return model, "l1"


def prune_fire_module(fire, amount, prev_keep=None):

    if prev_keep is not None:
        prune_conv_in(fire.squeeze, prev_keep)

    fire.expand1x1, keep1 = prune_conv_layer(
        fire.expand1x1, amount
    )

    fire.expand3x3, keep3 = prune_conv_layer(
        fire.expand3x3, amount
    )

    keep = torch.cat([keep1,keep3])

    return fire, keep


def prune_conv_layer(conv, amount):
    weight = conv.weight.data
    out_channels = weight.shape[0]

    #norma pruningu
    l1_norm=torch.sum(torch.abs(weight), dim=(1,2,3))

    num_prune = int(out_channels*amount)
    keep_idx = torch.topk(l1_norm, out_channels-num_prune)[1]

    new_weight=weight[keep_idx,:,:,:].clone()

    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=new_weight.shape[0],
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None)
    )

    new_conv.weight.data = new_weight

    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_idx].clone()

    return new_conv, keep_idx


def prune_conv_in(conv, keep_idx):
    conv.weight.data = conv.weight.data[:, keep_idx, :, :].clone()
    conv.in_channels = len(keep_idx)

class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset(index)
        img = sample['image']
        label = sample['label']
        if self.transform:
            img=self.transform
        return img, label
    
# --- 3. IterableDataset z ręcznym batchowaniem ---
class HFDatasetStream(IterableDataset):
    def __init__(self, hf_dataset, transform=None, limit=1000):
        self.dataset = hf_dataset
        self.transform = transform
        self.limit = limit

    def __iter__(self):
        batch_imgs, batch_labels = [], []
        count = 0
        for sample in self.dataset:
            if count >= self.limit:
                break

            img = sample['image']
            label = sample['label']

            # upewnij się, że obraz jest RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            if self.transform:
                img = self.transform(img)

            batch_imgs.append(img)
            batch_labels.append(label)
            count += 1

            # zwracaj batch po 32 obrazach
            if len(batch_imgs) == 32:
                yield torch.stack(batch_imgs), torch.tensor(batch_labels)
                batch_imgs, batch_labels = [], []

        # ostatni batch, jeśli jest niepełny
        if batch_imgs:
            yield torch.stack(batch_imgs), torch.tensor(batch_labels)

def fine_tune(model, device):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # SqueezeNet wymaga 224x224
        transforms.ToTensor(),          # zamiana PIL.Image → Tensor [C,H,W]
    ])

    ds_stream = load_dataset("mrm8488/ImageNet1K-train", split="train", streaming=True)
    train_dataset = HFDatasetStream(ds_stream, transform=transform, limit=1000)
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)  # batchy już w __iter__
    # 3. Optimizer i loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 4. Pętla treningowa (fine-tuning)
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            out = model(x_batch)
            out = out.view(out.size(0), -1)  # flatten [B, num_classes]
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()


    return model.eval()


def removeMasks(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, "weight")

    return model


def getPruning(num, model, amount):
    if num<5:
        if num==0:
            return pruning_random_unstruct(model, amount), "random_unstruct"
        elif num==1:
            return pruning_random_struct(model, amount), "random struct"
        elif num==2:
            return pruning_l1_unstruct(model,amount), "l1 unstruct"
        elif num==3:
            return pruning_l1_struct(model, amount), "l1 struct"
        elif num==4:
            return pruning_l2_struct(model, amount), "l2 struct"
    else:
        return model, "unprunned"


def checkSparsity(model):
    zeros=0
    total=0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            zeros += torch.sum(module.weight == 0)
            total += module.weight.nelement()
    sparsity = 100*zeros/total
    return sparsity
    
