# pip install pandas numpy matplotlib scikit-learn mplcursors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import mplcursors

# 设置中文字体（防止乱码）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 📥 读取数据
data = pd.read_csv('data.csv')  # 确保有 'x' 和 'y' 两列
x_raw = data[['x']].values
y_raw = data[['y']].values

# ⚙️ 标准化处理
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaled = scaler_x.fit_transform(x_raw)
y_scaled = scaler_y.fit_transform(y_raw).ravel()  # SVR 要求 y 为 1D

# ---------- 自动调参开关（默认关闭） ----------
# 将此值改为 True 可启用轻量级网格搜索自动调参（注意：对小数据集，cv 折数可能需要调整）
AUTO_TUNE = True
# ----------------------------------------------

# 📈 构建 SVR 模型
## kernel=核函数类型（'rbf'，'linear'，'poly'）
## C=100：拟合容忍度，越大越贴合数据
## epsilon=0.01：拟合精度控制，越小拟合越紧
model = SVR(kernel='rbf', C=50, epsilon=0.001)

if AUTO_TUNE:
    # 轻量级参数网格，避免过长时间搜索；可按需调整
    param_grid = {
        'kernel': ['rbf', 'poly', 'linear'],
        'C': [1, 10, 50, 100],
        'epsilon': [1e-3, 1e-2, 1e-1]
    }
    try:
        gs = GridSearchCV(SVR(), param_grid, cv=5, n_jobs=-1, scoring='r2')
        gs.fit(x_scaled, y_scaled)
        print(f"自动调参最佳参数: {gs.best_params_}")
        model = gs.best_estimator_
    except Exception as _e:
        # 如果自动调参失败（例如数据点少于 cv 折数），回退到默认手动参数
        print(f"自动调参失败，使用默认参数。错误: {_e}")

model.fit(x_scaled, y_scaled)

# 🔁 拟合结果反归一化
y_pred_scaled = model.predict(x_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
r2 = 1 - np.sum((y_raw - y_pred)**2) / np.sum((y_raw - np.mean(y_raw))**2)

# 📊 拟合曲线采样
x_plot = np.linspace(x_scaled.min(), x_scaled.max(), 200).reshape(-1, 1)
y_plot_scaled = model.predict(x_plot)
x_plot_orig = scaler_x.inverse_transform(x_plot)
y_plot_orig = scaler_y.inverse_transform(y_plot_scaled.reshape(-1, 1))

# 📍 拟合图像 + 鼠标悬停
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x_raw, y_raw, color='black', s=10, label='原始数据')
line, = ax.plot(x_plot_orig, y_plot_orig, color='green', label=f'SVR拟合 (R²={r2:.4f})')

cursor = mplcursors.cursor(line, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(
    f"x={sel.target[0]:.3f}\ny={sel.target[1]:.3f}"))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('SVR拟合曲线')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig('svr_fit.png')

# 📊 残差柱状图：真实值 - 拟合值
residuals = y_raw.flatten() - y_pred.flatten()

plt.figure(figsize=(10, 4))
plt.bar(range(len(residuals)), residuals, color='orange', width=0.6)
plt.xticks(ticks=range(len(x_raw)), labels=[f'{v[0]:.2f}' for v in x_raw], rotation=45)
plt.axhline(0, color='red', linestyle='--', label='零残差线')
plt.xlabel('数据点索引')
plt.ylabel('残差 (y真实 - y拟合)')
plt.title('SVR拟合残差柱状图')
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.savefig('svr_residuals_bar.png')
plt.show()

# 💾 提示是否保存拟合曲线点集
choice = input("是否保存 SVR 拟合曲线为 CSV？输入 y 保存，输入 n 跳过：").strip().lower()
if choice == 'y':
    pd.DataFrame({
        'x': x_plot_orig.flatten(),
        'y': y_plot_orig.flatten()
    }).to_csv('svr_fit_curve.csv', index=False)
    print("✅ 拟合曲线已保存为 svr_fit_curve.csv")
elif choice == 'n':
    print("⏭️ 已跳过保存")
else:
    print("⚠️ 无效输入，未保存任何内容")
