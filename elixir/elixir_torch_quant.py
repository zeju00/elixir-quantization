import torch
import torch.quantization as tq
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

# 모델 로드
model = torch.load('elixir_torch.pt')  # 전체 모델 로드

# 장치 설정
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)

# 양자화 준비
model.qconfig = tq.get_default_qat_qconfig('fbgemm')
tq.prepare_qat(model, inplace=True)

# Load the dataset
file_path = 'elixir_dataset/fl_default_of10.csv'
data = pd.read_csv(file_path)

# Separate input and output data
input_data = data[['switch_e', 'switch_c', 'host', 'connection', 'interval', 'link', 'hop']].values
output_data = data[['avgSentMsg', 'avgSentByte', 'secMaxSendMsg', 'secMaxSendByte']].values

# Convert to tensors
input_tensor = torch.tensor(input_data, dtype=torch.float32)
output_tensor = torch.tensor(output_data, dtype=torch.float32)

# Create dataset and dataloader
dataset = TensorDataset(input_tensor, output_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 재학습이므로 낮은 학습률 사용

# Training loop (running on GPU) - 재학습 진행
def train_gpu(model, dataloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        epoch_loss = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}')

# 재학습 수행
train_gpu(model, dataloader, criterion, optimizer, epochs=10)

# 양자화된 모델로 변환
model.eval()
model.cpu()
quantized_model = tq.convert(model, inplace=True)

# 양자화된 모델 저장
torch.save(quantized_model, 'elixir_torch_quantized.pt')

with torch.no_grad():  # Disable gradient calculation
    predictions = model(test_input_tensor)
# Calculate MSE between predictions and actual output
mse_loss = nn.MSELoss()
mse = mse_loss(predictions, test_output_tensor)

# Calculate R² (R-squared)
# Total Sum of Squares (proportional to the variance of the data)
ss_total = torch.sum((test_output_tensor - torch.mean(test_output_tensor, dim=0)) ** 2)

# Residual Sum of Squares
ss_residual = torch.sum((test_output_tensor - predictions) ** 2)

# R² calculation
r2_score = 1 - ss_residual / ss_total

# Convert the list of predictions to a numpy array
predictions_np = predictions.cpu().numpy()

# Convert predictions to a DataFrame
#predictions_np = pd.DataFrame(predictions, columns=['avgSentMsg', 'avgSentByte', 'secMaxSendMsg', 'secMaxSendByte'])
actual_np = test_output_tensor.cpu().numpy()

# Split the predictions and actual output into 'avg' and 'max' components
predictions_avg = predictions[:, :2]  # avgSentMsg, avgSentByte
predictions_max = predictions[:, 2:]  # secMaxSendMsg, secMaxSendByte

actual_avg = test_output_tensor[:, :2]  # avgSentMsg, avgSentByte
actual_max = test_output_tensor[:, 2:]  # secMaxSendMsg, secMaxSendByte

# Calculate MSE for 'avg' components
mse_avg = F.mse_loss(predictions_avg, actual_avg)

# Calculate MSE for 'max' components
mse_max = F.mse_loss(predictions_max, actual_max)

# Combine the predictions and actual values into a single DataFrame
comparison_df = pd.DataFrame({
    'Predicted_avgSentMsg': predictions_np[:, 0],
    'Actual_avgSentMsg': actual_np[:, 0],
    'Predicted_avgSentByte': predictions_np[:, 1],
    'Actual_avgSentByte': actual_np[:, 1],
    'Predicted_secMaxSendMsg': predictions_np[:, 2],
    'Actual_secMaxSendMsg': actual_np[:, 2],
    'Predicted_secMaxSendByte': predictions_np[:, 3],
    'Actual_secMaxSendByte': actual_np[:, 3]
})

# Print MSE
#print(f'Mean Squared Error (MSE) on test data: {mse.item():.4f}')

# Print the MSE values
#print(f'Mean Squared Error (MSE) for avg components: {mse_avg.item():.4f}')
#print(f'Mean Squared Error (MSE) for max components: {mse_max.item():.4f}')

# Preview the prediction results
print(comparison_df)

# Print the MSE values
print(f'Mean Squared Error (MSE) on test data: {mse.item():.4f}')
print(f'Mean Squared Error (MSE) for avg components: {mse_avg.item():.4f}')
print(f'Mean Squared Error (MSE) for max components: {mse_max.item():.4f}')
print(f'R² (R-squared) on test data: {r2_score.item():.4f}')

# Optionally, save the predictions to a CSV
#comparison_df.to_csv('predictions.csv', index=False)
