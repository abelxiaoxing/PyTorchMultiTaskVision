import torch
import onnx
import tensorrt as trt
from io import BytesIO
import torch.quantization as quant

def dynamic_quantize_model(model_weight_path, quantize_output_path, dtype, device):
    model = torch.load(model_weight_path, map_location=device,weights_only=False)["model"]
    if dtype == 8:
        quantized_model = quant.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    elif dtype == 16:
        quantized_model = quant.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.float16
        )
    else:
        raise ValueError("dtype must be either 'int8' or 'float16'")
    torch.save(quantized_model, quantize_output_path)
    print(f"Dynamic quantized model saved as {quantize_output_path}")


def pth2jit(
        model_weight_path="train_cls/output/checkpoint-best.pth", 
        jit_output_path="best_model.pth",
        device="cuda",
    ):
    checkpoint = torch.load(model_weight_path, map_location=device,weights_only=False) 
    model = checkpoint["model"]
    model.eval()
    input_shape = checkpoint["input_shape"]
    input = torch.rand(*input_shape).to(device)
    traced_model = torch.jit.trace(model, input)
    torch.jit.save(traced_model, jit_output_path)
    print(f"TorchScript model saved as {jit_output_path}")

def pth2onnx(
    model_weight_path="train_cls/output/checkpoint-best.pth",
    device="cuda",
    onnx_output_path="train_cls/output/checkpoint-best.onnx",
    simplify=False,
):
    # 加载模型
    checkpoint = torch.load(model_weight_path, map_location=device,weights_only=False)
    model = checkpoint["model"]
    model.eval()
    input_shape = checkpoint["input_shape"]
    input = torch.rand(*input_shape).to(device)
    # input = torch.rand(1, 3, 224, 224).cuda()
    torch.onnx.export(
        model,
        input,
        onnx_output_path,
        opset_version=10
    )
    # 检测onnx模型是否转换成功
    model_onnx = onnx.load(onnx_output_path)
    onnx.checker.check_model(model_onnx)
    if simplify:
        print(f"Simplifying with onnx-simplifier {onnxsim.__version__}.")
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, "assert check failed"
        onnx.save(model_onnx, onnx_output_path)

    print(f"Onnx model save as {onnx_output_path}")


def onnx2trt(onnx_model_weight_path, trt_output_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_model_weight_path)

    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        pass

    # 创建构建配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    serialized_engine = builder.build_serialized_network(network, config)
    with open(trt_output_path, "wb") as ff:
        ff.write(serialized_engine)

    print(f"TensorRT engine saved as {trt_output_path}")


def pth2onnx_in_memory(
    model_weight_path="train_cls/output/checkpoint-best.pth",
    device="cuda",
    simplify=False,
):
    # 加载PyTorch模型
    checkpoint = torch.load(model_weight_path, map_location=device,weights_only=False)
    model = checkpoint["model"]
    model.eval()
    input_shape = checkpoint["input_shape"]
    input = torch.rand(*input_shape).to(device)

    # 在内存中将模型导出为ONNX
    f = BytesIO()
    torch.onnx.export(model, input, f)
    onnx_model = onnx.load_model_from_string(f.getvalue())

    # 如果需要，简化ONNX模型
    if simplify:
        print(f"Simplifying with onnx-simplifier {onnxsim.__version__}.")
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, "Simplification failed"

    return onnx_model


def onnx2trt_in_memory(onnx_model, trt_output_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 从内存中解析ONNX模型
    success = parser.parse(onnx_model.SerializeToString())
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        raise RuntimeError("Failed to parse the ONNX model")

    # 创建构建器配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)

    # 构建TensorRT引擎
    serialized_engine = builder.build_serialized_network(network, config)
    with open(trt_output_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved as {trt_output_path}")


# 把onnx转换的model放在内存直接trt转换
def pth2trt(
    model_weight_path="train_cls/output/checkpoint-best.pth",
    device="cuda",
    trt_output_path="train_cls/output/checkpoint-best.trt",
    simplify=False,
):
    onnx_model = pth2onnx_in_memory(model_weight_path, device, simplify)
    onnx2trt_in_memory(onnx_model, trt_output_path)

def convert_model_ema_to_model(model_weight_path, output_path):
    checkpoint = torch.load(model_weight_path, map_location="cpu",weights_only=False)
    checkpoint["model"].load_state_dict(checkpoint["model_ema"])
    checkpoint.pop("model_ema", None)
    checkpoint.pop("optimizer", None)
    checkpoint.pop("scaler", None)
    torch.save(checkpoint, output_path)
    print(f"Converted checkpoint saved to: {output_path}")


if __name__ == "__main__":
    convert_model_ema_to_model("","")
