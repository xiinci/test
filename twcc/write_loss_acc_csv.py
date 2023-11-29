import csv
import os
from datetime import datetime
import numpy as np

# 模擬訓練模型的過程
def train_model(epochs, learning_rate):
    loss_data = []
    acc_data = []
    for epoch in range(epochs):
        # 使用簡單的函數模擬損失和準確率的變化
        loss = 1 / (epoch + 1) * learning_rate
        acc = 1 - loss

        loss_data.append((epoch, loss))
        acc_data.append((epoch, acc))
    
    return loss_data, acc_data

# 生成帶有當前日期時間的文件名
def generate_filename():
    current_time = datetime.now().strftime("%y%m%d_%H%M")
    return current_time

# 確保目錄存在，如果不存在則創建
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 將數據保存到CSV文件，並將文件放入相應的資料夾
def save_to_csv(folder_name, filename, data, fieldnames):
    ensure_dir(folder_name)  # 創建資料夾
    filepath = os.path.join(folder_name, filename)
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)  # 寫入表頭
        for row in data:
            writer.writerow(row)  # 寫入數據

# 設定訓練參數
EPOCHS = 500  # 訓練輪數
LEARNING_RATE = 0.01  # 學習率

# 訓練模型並獲取數據
loss_data, acc_data = train_model(EPOCHS, LEARNING_RATE)

# 生成文件名
current_time = generate_filename()
loss_filename = f"{current_time}_loss.csv"
acc_filename = f"{current_time}_acc.csv"

# 保存損失和準確率數據到CSV文件
save_to_csv('loss_acc', loss_filename, loss_data, ['epoch', 'loss'])
save_to_csv('loss_acc', acc_filename, acc_data, ['epoch', 'acc'])

print(f"損失數據已保存至: loss_acc/{loss_filename}")
print(f"準確率數據已保存至: loss_acc/{acc_filename}")
