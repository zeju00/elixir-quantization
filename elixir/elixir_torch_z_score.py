import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # Importing the tqdm library for progress bars
import numpy as np
from scipy.stats import zscore
import torch.nn.functional as F

# Load the data
file_path = 'elixir_dataset/fl_default_of10.csv'
data = pd.read_csv(file_path)

# Separate input and output data
input_data = data[['switch_e', 'switch_c', 'host', 'connection', 'interval', 'link', 'hop']]
output_data = data[['avgSentMsg', 'avgSentByte', 'secMaxSendMsg', 'secMaxSendByte']]

# Detect and remove outliers using Z-score
z_scores = np.abs(zscore(output_data))
threshold = 3  # Commonly used threshold value
non_outliers = (z_scores < threshold).all(axis=1)  # Keep rows where all columns are within the threshold

# Filter the data to remove outliers
input_data_clean = input_data[non_outliers]
output_data_clean = output_data[non_outliers]

# Convert to tensors
input_tensor = torch.tensor(input_data_clean.values, dtype=torch.float32)
output_tensor = torch.tensor(output_data_clean.values, dtype=torch.float32)

# Move tensors to GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)
output_tensor = output_tensor.to(device)

# Create dataset and dataloader (batch size 128)
dataset = TensorDataset(input_tensor, output_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Define the neural network model
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(7, 1024)  # 7 input nodes, 128 hidden nodes
        self.layer2 = nn.Linear(1024, 512)  # 64 hidden nodes
        self.layer3 = nn.Linear(512, 128)  # 32 hidden nodes
        self.layer4 = nn.Linear(128, 32)  # 16 hidden nodes
        self.layer5 = nn.Linear(32, 4)   # 4 output nodes (number of output variables)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x

# Initialize the model and move to GPU
model = ANNModel().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (running on GPU)
def train_gpu(model, dataloader, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        epoch_loss = 0
        mse_avgSentMsg = 0
        mse_avgSentByte = 0
        mse_secMaxSendMsg = 0
        mse_secMaxSendByte = 0

        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Calculate individual MSEs
            mse_avgSentMsg += nn.functional.mse_loss(outputs[:, 0], targets[:, 0], reduction='sum').item()
            mse_avgSentByte += nn.functional.mse_loss(outputs[:, 1], targets[:, 1], reduction='sum').item()
            mse_secMaxSendMsg += nn.functional.mse_loss(outputs[:, 2], targets[:, 2], reduction='sum').item()
            mse_secMaxSendByte += nn.functional.mse_loss(outputs[:, 3], targets[:, 3], reduction='sum').item()

        # Normalize by the number of samples in the dataset
        num_samples = len(dataloader.dataset)
        mse_avgSentMsg /= num_samples
        mse_avgSentByte /= num_samples
        mse_secMaxSendMsg /= num_samples
        mse_secMaxSendByte /= num_samples

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}')
        print(f'    MSE avgSentMsg: {mse_avgSentMsg:.4f}')
        print(f'    MSE avgSentByte: {mse_avgSentByte:.4f}')
        print(f'    MSE secMaxSendMsg: {mse_secMaxSendMsg:.4f}')
        print(f'    MSE secMaxSendByte: {mse_secMaxSendByte:.4f}')

# Train the model
train_gpu(model, dataloader, criterion, optimizer, epochs=100)

# Load the test dataset
test_file_path = 'elixir_dataset/fl_default_of10_scoring.csv'
test_data = pd.read_csv(test_file_path)

# Extract input data (from 'switch_e' to 'hop' in the test dataset)
test_input_data = test_data[['switch_e', 'switch_c', 'host', 'connection', 'interval', 'link', 'hop']].values

# Extract the actual output data (the target values for comparison)
test_output_data = test_data[['avgSentMsg', 'avgSentByte', 'secMaxSendMsg', 'secMaxSendByte']].values

# Convert to tensor and move to GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
test_input_tensor = torch.tensor(test_input_data, dtype=torch.float32).to(device)
test_output_tensor = torch.tensor(test_output_data, dtype=torch.float32).to(device)

# Perform prediction using the model
model.eval()  # Switch to evaluation mode (disables dropout, batch normalization)

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
print(f'Mean Squared Error (MSE) on test data: {mse.item():.4f}')

# Preview the prediction results
print(comparison_df)

# Split the predictions and actual output into 'avg' and 'max' components
predictions_avg = predictions[:, :2]  # avgSentMsg, avgSentByte
predictions_max = predictions[:, 2:]  # secMaxSendMsg, secMaxSendByte

actual_avg = test_output_tensor[:, :2]  # avgSentMsg, avgSentByte
actual_max = test_output_tensor[:, 2:]  # secMaxSendMsg, secMaxSendByte

# Calculate MSE for 'avg' components
mse_avg = F.mse_loss(predictions_avg, actual_avg)

# Calculate MSE for 'max' components
mse_max = F.mse_loss(predictions_max, actual_max)

# Print the MSE values
print(f'Mean Sqaured Error (MSE) on test data: {mse.item():.4f}')
print(f'Mean Squared Error (MSE) for avg components: {mse_avg.item():.4f}')
print(f'Mean Squared Error (MSE) for max components: {mse_max.item():.4f}')
#print(f'Mean Squared Error (MSE) on test data: {mse.item():.4f}')
print(f'R²(R-squared) on test data: {r2_score.item():.4f}')

# Optionally, save the predictions to a CSV file
#comparison_df.to_csv('result/predictions_z.csv', index=False)
