import torch
import netron
from lstm import LSTM

model = LSTM(input_size = 7 , hidden_size=16 , num_layers=2 , output_size=1 )
input = torch.randn(1 , 10 , 7)

torch.onnx.export(model , input , f = "lstm_model.onnx" , export_params=True)