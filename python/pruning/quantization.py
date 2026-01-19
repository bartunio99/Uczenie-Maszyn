import torch
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import HistogramObserver
from preparaData import prepare

def quant_model(img_dir, normal_model):
    calib_images, _ = prepare(img_dir, sampleSize=800)
    calib_images = calib_images.to("cpu")

    qconfig = torch.ao.quantization.QConfig(
        activation=HistogramObserver.with_args(reduce_range=True),
        weight=torch.ao.quantization.default_per_channel_weight_observer
    )
    qconfig_dict = {"": qconfig}

    example_inputs = torch.randn(1, 3, 224, 224)
    model_prepared = prepare_fx(
        normal_model,
        qconfig_dict,
        example_inputs=example_inputs
    )
    with torch.no_grad():
        for i in range(0, calib_images.size(0), 16):
            model_prepared(calib_images[i:i+16])

    model_int8 = convert_fx(model_prepared)
    model_int8.eval()
    return model_int8