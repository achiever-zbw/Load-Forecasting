import pandas as pd
import torch

# 从 excel 文件中获取输入与目标信息
def get_input_target_data(input_data_path , target_data_path) :
    input_data = pd.read_excel(input_data_path)
    target_data = pd.read_excel(target_data_path)["total_load"]

    return input_data , target_data


def load_data(features_train , target_train , device) :
    """转化为 Pytorch 张量"""
    X_train = torch.FloatTensor(features_train).to(device)
    y_train = torch.FloatTensor(target_train).unsqueeze(1).to(device)
    # 创建dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train , y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset , batch_size = 32 , shuffle = True)
    return train_loader