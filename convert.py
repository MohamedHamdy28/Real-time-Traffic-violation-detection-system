import torch
from utils.raft import RAFT
from argparse import Namespace
import onnx
import onnx_tf

args = Namespace(images_dir='test_video', output_dir='test_result')
model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(
    r'models/raft-things.pth', map_location="cuda"))
model.eval()
input_shape = (1, 3, 184, 320)
dummy_input1 = torch.randn(input_shape).to('cuda')
dummy_input2 = torch.randn(input_shape).to('cuda')
onnx_model_path = 'models/raft.onnx'
torch.onnx.export(model.module.to('cuda').eval(), (dummy_input1.to('cuda'), dummy_input2.to('cuda')),
                  onnx_model_path, input_names=['input1', 'input2'], verbose=False)

onnx_model = onnx.load(onnx_model_path)

# Convert the ONNX model to TensorFlow format
# tf_model_path = 'models/raft.pb'
# tf_rep = onnx_tf.backend.prepare(onnx_model)
# tf_rep.export_graph(tf_model_path)
