import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


# 1. 数据准备 - 从 Excel 文件读取多个 sheet
def read_excel_sheets(file_path):
    xls = pd.ExcelFile(file_path)
    df_list = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


# 读取所有 sheet 并合并
df = read_excel_sheets('9025.xlsx')  # 替换 'your_file.xlsx' 为你的文件名

# 给列命名
df.columns = ['A', 'B_附加值', 'B']

# 确保数据类型正确
df['A'] = df['A'].astype(str)
df['B_附加值'] = df['B_附加值'].astype(float)
df['B'] = df['B'].astype(float)

# 打印前几行来验证数据
print(df.head())
print(f"Total rows: {len(df)}")

# 2. 数据预处理
le = LabelEncoder()
df['A_encoded'] = le.fit_transform(df['A'])


# 3. 创建自定义数据集
class CustomDataset(Dataset):
    def __init__(self, B_附加值, B, A):
        self.B_附加值 = torch.FloatTensor(B_附加值)
        self.B = torch.FloatTensor(B)
        self.A = torch.LongTensor(A)

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        return self.B_附加值[idx], self.B[idx], self.A[idx]


# 4. 定义模型
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttentionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context_vector)
        return output


# 5. 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for B_附加值, B, A in train_loader:
            x = torch.stack([B_附加值, B], dim=1).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, A)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 6. 预测函数
def predict_A(model, B_附加值, B):
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor([[B_附加值, B]]).unsqueeze(1)
        output = model(x)
        predicted = torch.argmax(output, dim=1)
        return le.inverse_transform(predicted.numpy())[0]


# 7. 主程序
X = df[['B_附加值', 'B']].values
y = df['A_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = CustomDataset(X_train[:, 0], X_train[:, 1], y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

input_size = 2
hidden_size = 64
num_classes = len(le.classes_)

model = AttentionBiLSTM(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, num_epochs=100)

# 8. 测试预测
print(predict_A(model, 1, 2))
print(predict_A(model, 0.5, 3))