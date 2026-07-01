import onnx

# ONNX modelini yükle
model = onnx.load("yolov8s.onnx")

# Modelin girişlerini ve çıkışlarını yazdır
print("Model Inputs:")
for input in model.graph.input:
    print(f"{input.name} - Shape: {input.type.tensor_type.shape.dim}")

print("\nModel Outputs:")
for output in model.graph.output:
    print(f"{output.name} - Shape: {output.type.tensor_type.shape.dim}")

# Modeldeki tüm katmanları yazdır
print("\nModel Layers:")
for node in model.graph.node:
    print(f"Name: {node.name}, OpType: {node.op_type}, Outputs: {node.output}")

