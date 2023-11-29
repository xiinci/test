import csv, os, re, zipfile, torch
from flask import Flask, render_template, send_from_directory, redirect, url_for, request, jsonify


app = Flask(__name__)


@app.route('/')
def index():
    # 從本地 CSV 檔案讀取數據
    with open('project_list.csv', 'r', encoding='utf-8') as csvfile:
        # 創建 csv 讀取器
        reader = csv.DictReader(csvfile)
        # 將 csv 數據轉換為字典列表
        data = [row for row in reader]
        print(data)
    # 將數據傳遞給模板
    return render_template('index.html', table_data=data)

# 讀取並返回指定資料夾中的唯一日期時間檔案名稱
def list_files(directory):
    try:
        files = os.listdir(directory)
        unique_dates = set()
        for file in files:
            # 使用正則表達式移除 '_loss.csv' 或 '_acc.csv'
            match = re.match(r"(\d{6}_\d{4})_(train_loss|train_acc|val_loss|val_acc)\.csv", file)
            if match:
                unique_dates.add(match.group(1))
        return list(unique_dates)
    except FileNotFoundError:
        return []

# 讀loss_acc資料夾裡的csv檔######################################################################
@app.route('/project_detail')
def project_detail():
    # 從本地 CSV 檔案讀取數據
    with open('experiment_list.csv', 'r', encoding='utf-8') as csvfile:
        # 創建 csv 讀取器
        reader = csv.DictReader(csvfile)
        # 將 csv 數據轉換為字典列表
        data = [row for row in reader]
        print(data)
    
    # 獲取 loss 和 acc 資料夾中的檔案名稱
    loss_acc_files = list_files('loss_acc')

    # 將數據、檔案名稱傳遞給模板
    return render_template('project_detail.html', table_data=data, loss_acc_files=loss_acc_files)


# 讀圖表####
@app.route('/diagrams/<filename>')
def diagrams(filename):
    return send_from_directory('loss_acc_diagram', filename)

#新增實驗、上傳pt zip檔####################################
# @app.route('/add_experiment', methods=['POST'])
# def add_experiment():
#     # 确认是否收到文件
#     if 'modelFile' not in request.files or 'datasetFile' not in request.files:
#         return jsonify({'success': False, 'message': 'No file part'})

#     model_file = request.files['modelFile']
#     dataset_file = request.files['datasetFile']

#     # 确认文件是否有名字（即用户有没有选择文件）
#     if model_file.filename == '' or dataset_file.filename == '':
#         return jsonify({'success': False, 'message': 'No selected file'})

#     # 保存文件
#     model_path = os.path.join('uploads', model_file.filename)
#     dataset_path = os.path.join('uploads', dataset_file.filename)
#     model_file.save(model_path)
#     dataset_file.save(dataset_path)

#     # 加载模型
#     model = torch.jit.load(model_path)

#     # 解压数据集
#     with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
#         zip_ref.extractall('datasets')

#     # 假设数据集解压到了 datasets 文件夹
#     # 以下是假设的训练过程
#     # train_model(model, 'datasets/...') # 你需要实现 train_model 函数

#     # 假设 train_model 函数已经保存了一个 CSV 文件
#     # 返回成功消息和 CSV 文件的路径
#     return jsonify({'success': True, 'message': 'Experiment added successfully', 'data_path': 'path_to_csv'})


###############################################################################################
# 設定檔案上傳的目錄
UPLOAD_FOLDER = 'add_experiment'
# 允許上傳的檔案類型
ALLOWED_EXTENSIONS = {'pt', 'zip'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 確保檔案類型安全
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/add_experiment', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        model_file = request.files['modelFile']
        dataset_file = request.files['datasetFile']

        if model_file and dataset_file and allowed_file(model_file.filename) and allowed_file(dataset_file.filename):
            model_filename = secure_filename(model_file.filename)
            dataset_filename = secure_filename(dataset_file.filename)
            
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
            dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_filename)
            
            model_file.save(model_path)
            dataset_file.save(dataset_path)
            
            # 讀取模型
            model = torch.jit.load(model_path)
            
            # 解壓數據集
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                zip_ref.extractall('datasets')
            
            # 這裡會有您的訓練模型的程式碼
            # train_model(model, 'datasets/...')
            
            # 儲存訓練數據為 CSV 檔案
            # save_training_data_to_csv(training_data)
            
            # 返回結果頁面，顯示訓練結果
            return render_template('result.html', model_info=model_filename, dataset_info=dataset_filename)
        
        else:
            return jsonify({'error': 'Invalid file type'})

    # 如果不是 POST 請求，則渲染上傳表單
    return render_template('add_experiment.html')



if __name__ == '__main__':
    app.run(debug=True)
