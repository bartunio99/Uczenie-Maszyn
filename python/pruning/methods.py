#metody pruningu
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


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
    
