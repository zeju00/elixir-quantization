import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub, fuse_modules, prepare_qat, convert, quantize_dynamic, QConfig, default_observer, default_per_channel_weight_observer, get_default_qconfig, default_weight_observer
import torchvision
import torchvision.transforms as transforms
import time
from tqdm import tqdm
import pandas as pd
import contextlib
import random
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import psutil
import os
from torch.profiler import profile, record_function, ProfilerActivity

# Device setup
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(7, 1024)  # 7 input nodes, 128 hidden nodes
        self.layer2 = nn.Linear(1024, 512)  # 64 hidden nodes
        self.layer3 = nn.Linear(512, 128)  # 32 hidden nodes
        self.layer4 = nn.Linear(128, 32)  # 16 hidden nodes
        self.layer5 = nn.Linear(32, 4)   # 4 output nodes (number of output variables)
        
        # Quantization Stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        x = self.dequant(x)
        return x

# Load pre-trained model
model = torch.load('elixir_torch.pt')

# Load the data
file_path = 'elixir_dataset/fl_default_of10.csv'
data = pd.read_csv(file_path)

# Separate input and output data
input_data = data[['switch_e', 'switch_c', 'host', 'connection', 'interval', 'link', 'hop']].values
output_data = data[['avgSentMsg', 'avgSentByte', 'secMaxSendMsg', 'secMaxSendByte']].values

# Convert to tensors
input_tensor = torch.tensor(input_data, dtype=torch.float32)
output_tensor = torch.tensor(output_data, dtype=torch.float32)

# Move tensors to GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)
output_tensor = output_tensor.to(device)

# Create dataset and dataloader (batch size 64)
dataset = TensorDataset(input_tensor, output_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Load the test dataset
test_file_path = 'elixir_dataset/fl_default_of10_scoring.csv'
test_data = pd.read_csv(test_file_path)

# Extract input data (from 'switch_e' to 'hop' in the test dataset)
test_input_data = test_data[['switch_e', 'switch_c', 'host', 'connection', 'interval', 'link', 'hop']].values

# Extract the actual output data (the target values for comparison)
test_output_data = test_data[['avgSentMsg', 'avgSentByte', 'secMaxSendMsg', 'secMaxSendByte']].values

# Convert to tensor and move to GPU
#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
test_input_tensor = torch.tensor(test_input_data, dtype=torch.float32).to('cpu')
test_output_tensor = torch.tensor(test_output_data, dtype=torch.float32).to('cpu')

# Measure CPU Memory Usage
def measure_cpu_memory_usage(model, input_tensor,description):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True) as prof:
        with record_function(description):
            with torch.no_grad():
                _ = model(input_tensor)
    total_memory_usage = sum([item.cpu_memory_usage for item in prof.key_averages()])
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    print(f"Total Memory Usage: {total_memory_usage / (1024 ** 2):.2f} MB")

# Evaluate Original Model
test_input = torch.tensor(test_input_data, dtype=torch.float32).to(device)
test_output = torch.tensor(test_output_data, dtype=torch.float32).to(device)

model = torch.load("elixir_torch.pt", map_location=device)
model.eval()

with torch.no_grad():
    predictions = model(test_input)

mse_loss = nn.MSELoss()
mse = mse_loss(predictions, test_output)
ss_residual = torch.sum((test_output - predictions) ** 2)
ss_total = torch.sum((test_output - torch.mean(test_output, dim=0)) ** 2)
r2_score = 1 - ss_residual / ss_total

predictions_np = predictions.cpu().numpy()
actual_np = test_output.cpu().numpy()

comparison_df = pd.DataFrame({
    'Original_Predicted_avgSentMsg': predictions_np[:, 0],
    'Actual_avgSentMsg': actual_np[:, 0],
    'Original_Predicted_avgSentByte': predictions_np[:, 1],
    'Actual_avgSentByte': actual_np[:, 1],
    'Original_Predicted_secMaxSendMsg': predictions_np[:, 2],
    'Actual_secMaxSendMsg': actual_np[:, 2],
    'Original_Predicted_secMaxSendByte': predictions_np[:, 3],
    'Actual_secMaxSendByte': actual_np[:, 3]
})

comparison_df.to_csv('result/original_predictions.csv', index=False)
print("Original predictions saved to 'original_predictions.csv'")

model.to('cpu')
measure_cpu_memory_usage(model, test_input_tensor, "Original Model Inference")
#print(f'Original Model\'s CPU Memory Usage: {original_cpu_memory_before:.2f}, {original_cpu_memory_after:.2f} MB')

# Perform Quantization
print("Starting Quantization...")

# Load the original pre-trained model
original_model = torch.load("elixir_torch.pt", map_location=device)

# PTQ: Post-Training Quantization
ptq_model = torch.load("elixir_torch.pt", map_location='cpu')

# Calibration (Static Quantization)
ptq_model.eval()

# PTQ qconfig
ptq_qconfig = QConfig(
        activation=default_observer.with_args(dtype=torch.quint8),
        weight=default_observer.with_args(dtype=torch.qint8)
        )

#ptq_model.fuse_model()  # Fuse layers
ptq_model.qconfig = ptq_qconfig
torch.quantization.prepare(ptq_model, inplace=True)

start_calibration = time.time()
calibration_loader = torch.utils.data.DataLoader(input_tensor, batch_size=128, shuffle=False)

# Use the test dataset for calibration
with torch.no_grad():
    for data in tqdm(calibration_loader, desc='PTQ Calibration', unit='batch'):
        data = data.to('cpu')
        ptq_model(data)
end_calibration = time.time()
print(f'Calibration Time: {end_calibration - start_calibration:.2f} seconds')

torch.quantization.convert(ptq_model, inplace=True)

with open('ptq_model_parameters.txt', 'w') as f:
    state_dict = ptq_model.state_dict()
    for name, param in state_dict.items():
        f.write(f"Name: {name}\n")
        #f.write(f"Shape: {param.shape}\n")
        #f.write(f"Device: {param.device}\n")
        #f.write(f"Requires Grad: {param.requires_grad}\n")
        #f.write(f"Data Type: {param.dtype}\n")
        f.write(f"Value: {param}\n\n")

# QAT: Quantization-Aware Training
qat_model = torch.load("elixir_torch.pt", map_location=device)
qat_model.eval()
#qat_model.fuse_model()  # Fuse the layers before training
qat_model.train()  # Switch back to train mode for QAT

qat_qconfig = torch.quantization.QConfig(
    activation=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine
    ),
    weight=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine
    )
)

qat_model.qconfig = qat_qconfig
torch.quantization.prepare_qat(qat_model, inplace=True)

optimizer = optim.Adam(qat_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 2 epoch fine-tuning for QAT
for i in tqdm(range(150), desc='Epoch'):
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = qat_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

qat_model.eval()
qat_model.to('cpu')  # Move to CPU before quantization
convert(qat_model, inplace=True)

with open('qat_model_parameters.txt', 'w') as f:
    state_dict = qat_model.state_dict()
    for name, param in state_dict.items():
        f.write(f"Name: {name}\n")
        #f.write(f"Shape: {param.shape}\n")
        #f.write(f"Device: {param.device}\n")
        #f.write(f"Requires Grad: {param.requires_grad}\n")
        #f.write(f"Data Type: {param.dtype}\n")
        f.write(f"Value: {param}\n\n")

print("Quantization completed. Starting evaluation iterations...")

#torch.backends.quantized.engine = 'fbgemm'

# Reload the original pre-trained model for each iteration
original_model = torch.load("elixir_torch.pt", map_location='cpu')
original_model.eval()

ptq_model.eval()
ptq_model.to('cpu')
test_input_tensor.to('cpu')

# PTQ 모델로 테스트 데이터 예측
with torch.no_grad():
    ptq_predictions = ptq_model(test_input_tensor)

# PTQ 모델 MSE 및 R² 계산
mse_loss = nn.MSELoss()
ptq_mse = mse_loss(ptq_predictions, test_output_tensor)
ptq_ss_residual = torch.sum((test_output_tensor - ptq_predictions) ** 2)
ss_total = torch.sum((test_output_tensor - torch.mean(test_output_tensor, dim=0)) ** 2)
ptq_r2_score = 1 - ptq_ss_residual / ss_total

# PTQ 예측 결과를 numpy로 변환
ptq_predictions_np = ptq_predictions.cpu().numpy()
actual_np = test_output_tensor.cpu().numpy()

# PTQ 예측 결과를 데이터프레임으로 생성
ptq_comparison_df = pd.DataFrame({
    'PTQ_Predicted_avgSentMsg': ptq_predictions_np[:, 0],
    'Actual_avgSentMsg': actual_np[:, 0],
    'PTQ_Predicted_avgSentByte': ptq_predictions_np[:, 1],
    'Actual_avgSentByte': actual_np[:, 1],
    'PTQ_Predicted_secMaxSendMsg': ptq_predictions_np[:, 2],
    'Actual_secMaxSendMsg': actual_np[:, 2],
    'PTQ_Predicted_secMaxSendByte': ptq_predictions_np[:, 3],
    'Actual_secMaxSendByte': actual_np[:, 3]
})

# PTQ 결과를 CSV 파일로 저장
ptq_comparison_df.to_csv('result/ptq_predictions.csv', index=False)
print("PTQ predictions saved to 'ptq_predictions.csv'")

# PTQ Model's CPU Memory Usage
measure_cpu_memory_usage(ptq_model, test_input_tensor, "PTQ Model Inference")
#print(f'PTQ Model\'s CPU Memory Usage: {ptq_cpu_memory_before:.2f}, {ptq_cpu_memory_after:.2f} MB')

# QAT 모델로 테스트 데이터 예측
qat_model.to('cpu')
qat_model.eval()  # 모델을 평가 모드로 설정
with torch.no_grad():
    qat_predictions = qat_model(test_input_tensor)

# QAT 모델 MSE 및 R² 계산
qat_mse = mse_loss(qat_predictions, test_output_tensor)
qat_ss_residual = torch.sum((test_output_tensor - qat_predictions) ** 2)
qat_r2_score = 1 - qat_ss_residual / ss_total

# QAT 예측 결과를 numpy로 변환
qat_predictions_np = qat_predictions.cpu().numpy()

# QAT 예측 결과를 데이터프레임으로 생성
qat_comparison_df = pd.DataFrame({
    'QAT_Predicted_avgSentMsg': qat_predictions_np[:, 0],
    'Actual_avgSentMsg': actual_np[:, 0],
    'QAT_Predicted_avgSentByte': qat_predictions_np[:, 1],
    'Actual_avgSentByte': actual_np[:, 1],
    'QAT_Predicted_secMaxSendMsg': qat_predictions_np[:, 2],
    'Actual_secMaxSendMsg': actual_np[:, 2],
    'QAT_Predicted_secMaxSendByte': qat_predictions_np[:, 3],
    'Actual_secMaxSendByte': actual_np[:, 3]
})

# QAT 결과를 CSV 파일로 저장
qat_comparison_df.to_csv('result/qat_predictions.csv', index=False)
print("QAT predictions saved to 'qat_predictions.csv'")

# QAT Model's CPU Memory Usage
measure_cpu_memory_usage(qat_model, test_input_tensor, "QAT Model Inference")
#print(f'QAT Model\'s CPU Memory Usage: {qat_cpu_memory_before:.2f}, {qat_cpu_memory_after:.2f} MB')

