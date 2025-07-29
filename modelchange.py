import torch
import onnx
import tensorrt as trt
from io import BytesIO
import torch.quantization as quant
import onnxsim

class ModelConverter:
    def __init__(self, model_weight_path: str, device: str = "cuda"):
        self.device = device
        checkpoint = torch.load(model_weight_path, map_location=self.device, weights_only=False)
        self.model = checkpoint["model"]
        self.model.eval()
        self.input_shape = checkpoint.get("input_shape")


    def dynamic_quantize(self, quantize_output_path: str, dtype: int):
        if dtype == 8:
            quantized_model = quant.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        elif dtype == 16:
            quantized_model = quant.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.float16)
        else:
            raise ValueError("dtype must be 8 (for int8) or 16 (for float16)")
        
        torch.save(quantized_model, quantize_output_path)
        print(f"动态量化模型已保存为 {quantize_output_path}")

    def to_jit(self, jit_output_path: str):
        scripted_model = torch.jit.script(self.model)
        torch.jit.save(scripted_model, jit_output_path)
        print(f"TorchScript模型已保存为 {jit_output_path}")

    def to_onnx(self, onnx_output_path = None, simplify = False) :
        dummy_input = torch.rand(*self.input_shape).to(self.device)
        
        f = BytesIO()
        torch.onnx.export(self.model, dummy_input, f, opset_version=10)
        onnx_model = onnx.load_model_from_string(f.getvalue())
        onnx.checker.check_model(onnx_model)

        if simplify:
            print(f"使用 onnx-simplifier {onnxsim.__version__} 进行简化。")
            onnx_model, check = onnxsim.simplify(onnx_model)
            if not check:
                raise RuntimeError("ONNX model simplification failed.")

        if onnx_output_path:
            onnx.save(onnx_model, onnx_output_path)
            print(f"ONNX model saved to {onnx_output_path}")
            return None
        
        return onnx_model

    def to_trt(self, trt_output_path: str, simplify_onnx: bool = False):
        print("Converting to ONNX in memory...")
        onnx_model = self.to_onnx(simplify=simplify_onnx)
        print("Building TensorRT engine...")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        if not parser.parse(onnx_model.SerializeToString()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model.")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
        
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine.")

        with open(trt_output_path, "wb") as f:
            f.write(serialized_engine)
        print(f"TensorRT engine saved to {trt_output_path}")


def convert_model_ema_to_model(model_weight_path, output_path):
    checkpoint = torch.load(model_weight_path, map_location="cpu", weights_only=False)
    checkpoint["model"].load_state_dict(checkpoint["model_ema"])
    checkpoint.pop("model_ema", None)
    checkpoint.pop("optimizer", None)
    checkpoint.pop("scaler", None)
    torch.save(checkpoint, output_path)
    print(f"转换后的检查点已保存到：{output_path}")


if __name__ == "__main__":
    
    # print("--- 正在转换 EMA 模型 ---")
    # convert_model_ema_to_model("train_cls/output/checkpoint-best.pth", "train_cls/output/output.pth")
    
    print("\n--- 正在初始化 ModelConverter ---")
    converter = ModelConverter(model_weight_path="train_cls/output/checkpoint-best.pth", device="cuda")
    
    print("\n--- 正在转换为 TorchScript (JIT) ---")
    converter.to_jit("train_cls/output/model.jit.pt")
    
    print("\n--- 正在转换为 ONNX ---")
    converter.to_onnx("train_cls/output/model.onnx", simplify=True)
    
    print("\n--- 正在转换为 TensorRT ---")
    converter.to_trt("train_cls/output/model.trt", simplify_onnx=True)
    
    print("\n--- 正在应用动态量化 (int8) ---")
    converter.dynamic_quantize("train_cls/output/model.qint8.pth", dtype=8)
        
    
    print("重构完成。")