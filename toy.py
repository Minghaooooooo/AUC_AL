from loss import *

import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


# 定义训练数据和标签
inputs = torch.randn(32, 10)  # 32个样本，每个样本有10个特征
labels = torch.randint(0, 2, (32, 5)).float()  # 32个样本的标签，每个样本有5个类别

# 初始化模型、损失函数和优化器
model = SimpleModel()
criterion = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 进行训练
for epoch in range(100):
    optimizer.zero_grad()  # 清零梯度

    outputs = model(inputs)
    loss = criterion(outputs, labels)

    loss.backward()  # 计算梯度
    optimizer.step()  # 更新模型参数

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# 在测试时，关闭梯度追踪
with torch.no_grad():
    test_inputs = torch.randn(10, 10)  # 10个测试样本
    test_outputs = model(test_inputs)
    print("Test Outputs:", test_outputs)
