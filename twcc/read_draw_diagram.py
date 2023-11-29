import matplotlib.pyplot as plt
import csv
import os

def read_csv_data(filepath):
    epochs = []
    values = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳過表頭
        for row in csv_reader:
            epochs.append(int(row[0]))
            values.append(float(row[1]))
    return epochs, values

def plot_data(epochs, values, title, filename):
    plt.figure()
    plt.plot(epochs, values, marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def process_files_in_directory(directory, output_directory):
    # 確保輸出目錄存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            epochs, values = read_csv_data(file_path)
            plot_title = 'Loss' if 'loss' in filename else 'Accuracy'
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_file_path = os.path.join(output_directory, output_filename)
            plot_data(epochs, values, plot_title, output_file_path)
            print(f"圖表已保存至: {output_file_path}")

# 指定資料夾路徑
input_directory = 'loss_acc'
output_directory = 'loss_acc_diagram'

# 處理資料夾中的文件
process_files_in_directory(input_directory, output_directory)
