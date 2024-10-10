import torch
import torch.nn as nn

# 모델 클래스 정의 (저장할 때 사용한 것과 동일해야 함)
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(7, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.layer4 = nn.Linear(128, 32)
        self.layer5 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x

# 저장된 모델 로드
model = torch.load('elixir_torch.pt')

# 파일을 열고 파라미터 정보를 저장
with open('model_parameters.txt', 'w') as f:
    for name, param in model.named_parameters():
        f.write(f"Name: {name}\n")
        f.write(f"Shape: {param.shape}\n")
        f.write(f"Device: {param.device}\n")
        f.write(f"Requires Grad: {param.requires_grad}\n")
        f.write(f"Data Type: {param.dtype}\n")
        f.write(f"Value: {param}\n\n")

print("모델 파라미터 정보가 'model_parameters.txt' 파일에 저장되었습니다.")
