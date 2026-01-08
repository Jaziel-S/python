# pip install numpy pandas matplotlib scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 设置中文字体（防止乱码）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv('data.csv')
x = data['x'].values
y = data['y'].values

# 定义拟合函数
def linear(x, a, b): return a * x + b
def quadratic(x, a, b, c): return a * x**2 + b * x + c
def cubic(x, a, b, c, d): return a * x**3 + b * x**2 + c * x + d
def exponential(x, a, b): return a * np.exp(b * x)
def logarithmic(x, a, b): return a * np.log(x) + b
def power(x, a, b): return a * x**b

# 拟合函数列表
functions = {
    '一次线性': linear,
    '二次多项式': quadratic,
    '三次多项式': cubic,
    '指数函数': exponential,
    '对数函数': logarithmic,
    '幂函数': power
}

# 📊 总对比图
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', s=10, label='原始数据')

for name, func in functions.items():
    try:
        if name in ['对数函数', '幂函数'] and np.any(x <= 0):
            print(f"跳过{name}：x 包含非正值")
            continue

        popt, _ = curve_fit(func, x, y)
        y_pred = func(x, *popt)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

        # 构造公式字符串
        param_str = ', '.join([f'{chr(97+i)}={v:.3f}' for i, v in enumerate(popt)])
        formula_map = {
            '一次线性': f'y = a·x + b\n{param_str}',
            '二次多项式': f'y = a·x² + b·x + c\n{param_str}',
            '三次多项式': f'y = a·x³ + b·x² + c·x + d\n{param_str}',
            '指数函数': f'y = a·exp(b·x)\n{param_str}',
            '对数函数': f'y = a·ln(x) + b\n{param_str}',
            '幂函数': f'y = a·x^b\n{param_str}'
        }

        # 添加到总图
        plt.plot(x, y_pred, label=f'{name} (R²={r2:.4f})')

        # 📁 单独图像保存
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, color='black', s=10, label='原始数据')
        plt.plot(x, y_pred, color='blue', label=f'{name}拟合 (R²={r2:.4f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{name}拟合图像')
        plt.grid(True)
        plt.legend()

        # 添加公式文本
        plt.text(0.5, -0.15, formula_map[name], transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.savefig(f'{name}_fit.png')
        plt.close()

    except Exception as e:
        print(f"{name} 拟合失败：{e}")

# 🎯 显示总图
plt.xlabel('x')
plt.ylabel('y')
plt.title('多种拟合方法对比')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('总拟合对比图.png')
plt.show()
