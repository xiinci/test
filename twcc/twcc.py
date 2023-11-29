from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import torch.utils.data as data
from torch.utils.data import Subset
from copy import deepcopy
import csv
from datetime import datetime

################################# 讀取yaml ################################
import yaml

with open('test.yaml', 'r') as f:
    data = yaml.safe_load(f)



################################# 資料讀取 ################################
# 讀取訓練資料參數設定
batch_size = data['TRAIN']['BATCH_SIZE']
shuffle  =data['TRAIN']['SHUFFLE']
input_image_size = data['TRAIN']['INPUT_IMAGE_SIZE']
data_transform = transforms.Compose([
        transforms.Resize(input_image_size),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomResizedCrop(input_image_size),
        transforms.ToTensor()
    ])
train_data = torchvision.datasets.CIFAR10(
    root='./cifar',
    train=True,
    transform=data_transform,
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root='./cifar',
    train=False,
    transform=data_transform,
    download=True
)

train ,val = torch.utils.data.random_split(dataset = train_data , lengths = [int(len(train_data)*0.8),int(len(train_data)*0.2)])
dataset_sizes = {'train':len(train) , 'val': len(val)}


# 將 train 、val、test data轉為 dataloader
train  = torch.utils.data.DataLoader(train,batch_size,shuffle,num_workers = 4)
val = torch.utils.data.DataLoader(val,batch_size ,shuffle,num_workers = 4)
test = torch.utils.data.DataLoader(test_data,batch_size,num_workers = 4)


###############################  定義模型訓練函式 ######################################

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    train_loss_data = []
    train_acc_data = []
    val_loss_data =[]
    val_acc_data = []
    since = time.time()


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                dataloaders = train
            else:
                dataloaders = val
            for inputs, labels in dataloaders:

                inputs = inputs.to(device)
                labels = labels.to(device)


                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':

                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                train_loss_data.append([epoch,epoch_loss])
                train_acc_data.append([epoch,epoch_acc.tolist()])
            if phase == 'val':
                val_loss_data.append([epoch,epoch_loss])
                val_acc_data.append([epoch,epoch_acc.tolist()])



            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    print('Best val Acc: {:4f}'.format(best_acc))


    model.load_state_dict(best_model_wts)
    return model,train_loss_data,train_acc_data,val_loss_data,val_acc_data

############################ csv 檔案設定函式 ####################################

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

############################# model 訓練參數設定 #################################
# 導入模型
model = torch.jit.load("model_scripted.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

momentum = False # 使用 sgd 會用到
alpha  = False # 使用 rmsprop 會用到

# 參數設定
# epoch設定
epochs = data['TRAIN']['EPOCHS']

# loss 設定
if data['TRAIN']['LOSS'] =='CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
else :
    criterion  = nn.MSELoss()

# 學習率相關參數設定
lr = data['TRAIN']['LR']
step_size  = data['TRAIN']['STEP_SIZE']
gamma  = data['TRAIN']['GAMMA']

# 確認是否有特殊參數
if data['TRAIN']['MOMENTUM'] :
    momentum = data['TRAIN']['MOMENTUM']
if 'ALPHA' in data :
    alpha = data['TRAIN']['ALPHA']

# 設定optimizer
if data['TRAIN']['OPTIMIZER'] == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr,momentum)
if data['TRAIN']['OPTIMIZER'] == 'RMSPROP' :
    optimizer = optim.RMSprop(model.parameters(), lr,alpha)

# 微調學習率
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

########################## model訓練 ##########################################

# 接收訓練完的模型以及csv所需之資料
model_ft,train_loss_data,train_acc_data,val_loss_data,val_acc_data = train_model(model, criterion, optimizer, exp_lr_scheduler, epochs)


########################## 驗證 - 使用test data ################################################
def evaluate(model,dataloader):
    model.eval()
    correct = 0
    total  = len(dataloader.dataset)
    for x,y in dataloader:
        x,y  =x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct+=torch.eq(pred,y).sum().float().item()
    return correct / total

# 驗證準確率
test_acc  = evaluate(model_ft,test)
print()
print('#'*45)
print('\ntest accuracy : ',test_acc)

###################################### 輸出 csv檔 ############################################

# 生成文件名
current_time = generate_filename()
train_loss_filename = f"{current_time}_train_loss.csv"
train_acc_filename = f"{current_time}_train_acc.csv"
val_loss_filename = f"{current_time}_val_loss.csv"
val_acc_filename = f"{current_time}_val_acc.csv"
# 保存損失和準確率數據到CSV文件
save_to_csv('loss_acc', train_loss_filename, train_loss_data, ['epoch', 'loss'])
save_to_csv('loss_acc', train_acc_filename, train_acc_data, ['epoch', 'acc'])
save_to_csv('loss_acc', val_loss_filename, val_loss_data, ['epoch', 'loss'])
save_to_csv('loss_acc', val_acc_filename, val_acc_data, ['epoch', 'acc'])
print(f"\n\n{train_loss_data}\n\n")

print(f"損失數據已保存至: loss_acc/{train_loss_filename}")
print(f"準確率數據已保存至: loss_acc/{train_acc_filename}")
print(f"損失數據已保存至: loss_acc/{val_loss_filename}")
print(f"準確率數據已保存至: loss_acc/{val_acc_filename}")
