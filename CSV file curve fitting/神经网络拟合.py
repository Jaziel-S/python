# pip install torch pandas matplotlib scikit-learn mplcursors

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import mplcursors

# 设置中文字体（防止乱码）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取 CSV 数据
data = pd.read_csv('data.csv')  # 确保文件中有 'x' 和 'y' 两列
x_raw = data[['x']].values
y_raw = data[['y']].values

# 数据标准化（避免尺度不一致）
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x = torch.tensor(scaler_x.fit_transform(x_raw), dtype=torch.float32)
y = torch.tensor(scaler_y.fit_transform(y_raw), dtype=torch.float32)

# 定义神经网络结构
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = Net()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练网络
loss_history = []
# 设置早停参数
patience = 500  # 容忍周期（例如连续500轮无明显改善）
min_delta = 1e-6  # 最小改善幅度
best_loss = float('inf')
trigger_times = 0

for epoch in range(5000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_value = loss.item()
    loss_history.append(loss_value)

    # 打印
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value:.6f}")

    # 早停判断
    if best_loss - loss_value > min_delta:
        best_loss = loss_value
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"📉 Loss在 {patience} 轮内无显著改善，提前停止训练于 Epoch {epoch}")
            break

# 用训练集 x 预测并反归一化
y_pred_scaled_train = model(x).detach().numpy()
y_pred_train = scaler_y.inverse_transform(y_pred_scaled_train)
y_true = y_raw

# 正确计算 R²
r2 = 1 - np.sum((y_true - y_pred_train)**2) / np.sum((y_true - np.mean(y_true))**2)

# 生成拟合曲线用于绘图，如果希望选点更精细，可以把 x_plot 的采样密度提高
x_plot = torch.linspace(x.min(), x.max(), 200).reshape(-1, 1)
y_plot_scaled = model(x_plot).detach().numpy()
x_plot_orig = scaler_x.inverse_transform(x_plot.numpy())
y_plot_orig = scaler_y.inverse_transform(y_plot_scaled)

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x_raw, y_raw, color='black', s=10, label='原始数据')
line, = ax.plot(x_plot_orig, y_plot_orig, color='red', label=f'神经网络拟合 (R²={r2:.4f})')

# 添加鼠标选点功能（只作用于拟合曲线）
cursor = mplcursors.cursor(line, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(
    f"x={sel.target[0]:.3f}\ny={sel.target[1]:.3f}"))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('拟合曲线mplcursors')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig('nn_fit.png')  # 可选：保存图像

# 绘制loss曲线，查看是否收敛
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练过程中的loss曲线')
plt.grid(True)
plt.tight_layout()
plt.show()

# 📢 提示用户是否保存拟合结果
choice = input("是否保存拟合曲线点集为 CSV？输入 y 保存，输入 n 跳过：").strip().lower()

if choice == 'y':
    pd.DataFrame({
        'x': x_plot_orig.flatten(),
        'y': y_plot_orig.flatten()
    }).to_csv('nn_fit_curve.csv', index=False)
    print("✅ 拟合曲线已保存为 nn_fit_curve.csv")
elif choice == 'n':
    print("⏭️ 已跳过保存")
else:
    print("⚠️ 无效输入，未保存任何内容")
