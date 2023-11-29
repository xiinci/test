import numpy as np
import matplotlib.pyplot as plt

# 設定 epoch 數量
epochs = 200
# 假設初始損失值
initial_loss = 5.0
# 假設損失值下降的速度
decay = 0.01

# 生成模擬的損失數據
np.random.seed(0)  # 確保每次生成的隨機數據相同
loss_data = initial_loss * np.exp(-decay * np.arange(epochs)) + np.random.normal(0, 0.1, epochs)

# 繪製損失數據的圖表
plt.figure(figsize=(10, 5))
plt.plot(loss_data, label='Training Loss')
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
