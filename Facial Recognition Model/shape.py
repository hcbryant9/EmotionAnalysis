import onnx

onnx_model = onnx.load('emotion-classifier-model.onnx')

print(onnx_model.graph.input[0].type.tensor_type.shape)