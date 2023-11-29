import csv
import random

# 模擬的數據生成函數
def create_mock_data(num_records):
    models = ['Model A', 'Model B', 'Model C', 'Model D', 'Model E']
    datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4', 'Dataset 5']
    status_list = ['完成', '執行中', '失敗']
    scores = [str(random.randint(80, 100)) for _ in range(num_records)]
    times = [f"{random.randint(1, 3)}小時" for _ in range(num_records)]
    
    data = []
    for i in range(num_records):
        data.append({
            "id": i+1,
            "模型名稱": random.choice(models),
            "資料集": random.choice(datasets),
            "狀態": random.choice(status_list),
            "時間": times[i],
            "分數": scores[i]
        })
    return data

# 生成數據
data = create_mock_data(10)

# CSV 檔案的檔名
csv_filename = 'experiment_list.csv'

# 寫入 CSV 檔案
with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.DictWriter(file, fieldnames=data[0].keys())
    writer.writeheader()  # 寫入標題
    writer.writerows(data)  # 寫入數據行
