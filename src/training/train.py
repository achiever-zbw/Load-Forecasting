import os
import numpy as np
import torch

from src.models.lstm import LSTM
from src.models.fusion_model import FusionModel
from src.data.data_loader import get_input_target_data , load_data
from src.data.preprocessing import create_sequences
from src.utils.time_utils import time_to_minutes
from sklearn.preprocessing import StandardScaler

data_dir = "/Users/zhaobowen/地铁空调负荷预测优化项目/data/raw/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备 : {device} ")

# 获取输入和目标值
input_data_dir , target_data_dir = os.path.join(data_dir , "一个月数据总表.xlsx") , os.path.join(data_dir , "一个月负荷数据总表.xlsx")
input_data , target_data = get_input_target_data(input_data_dir , target_data_dir)
# print(input_data.head())
# print(target_data.head())

# 时间序列样本构建
time_step = 10

# 在输入数据中加入时间列
input_data["minutes"] = input_data["time"].apply(time_to_minutes)

# 七个输入特征 时间、乘客流量、结构负荷、通风负荷、温度、湿度、设备数量
feature_columns = ["minutes" , "passengers" , "structure_load" , "vent_load" , "temp" , "hum" , "equip_num"]
features_data = input_data[feature_columns].values

train_ratio = 0.7 # 训练集
val_ratio = 0.1   # 验证集
test_ratio = 0.2  # 测试集

# 总数据量
total_samples = len(features_data)
train_size = int(train_ratio * total_samples)
val_size = int(val_ratio * total_samples)
test_size = total_samples - train_size - val_size
print(f"数据分割情况 : 训练集 = {train_size} , 验证集 = {val_size} , 测试集 = {test_size}")  # (201 , 28 , 59)

# 特征值 -- 训练、验证、测试的分割
features_train = features_data[:train_size]
features_val = features_data[train_size : train_size + val_size]
features_test = features_data[train_size + val_size : ]
# 目标值 -- 训练、验证、测试的分割
target_train = target_data.values[:train_size]
target_val = target_data.values[train_size : train_size + val_size]
target_test = target_data.values[train_size + val_size : ]

# 标准化
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_val_scaled = scaler.transform(features_val)
features_test_scaled = scaler.transform(features_test)

target_scaler = StandardScaler()
targets_train_scaled = target_scaler.fit_transform(target_train.reshape(-1, 1)).flatten()
targets_val_scaled = target_scaler.transform(target_val.reshape(-1, 1)).flatten()
targets_test_scaled = target_scaler.transform(target_test.reshape(-1, 1)).flatten()

print(f"标准化后数据形状: 训练集 = {features_train_scaled.shape} , 验证集 = {features_val_scaled.shape} , 测试集 = {features_test.shape}")

# 构建序列
X_train , y_train = create_sequences(features_train_scaled , targets_train_scaled , time_step)
X_val , y_val = create_sequences(features_val_scaled , targets_val_scaled , time_step)
X_test, y_test = create_sequences(features_test_scaled, targets_test_scaled, time_step)

print(f"序列数据形状: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"序列数据形状: X_val = {X_val.shape}, y_val = {y_val.shape}")
print(f"序列数据形状: X_test = {X_test.shape}, y_test = {y_test.shape}")

print(f"训练集: X = {X_train.shape}, y = {y_train.shape}")
print(f"验证集: X = {X_val.shape}, y = {y_val.shape}")
print(f"测试集: X = {X_test.shape}, y = {y_test.shape}")



# 获取 Loader 后的数据
train_loader = load_data(X_train , y_train , device)

# 训练
def train(model , loss_fn , optimizer , scheduler , train_dataloader , X_val , y_val , epochs = 200) :
    model.to(device).train()

    train_loss = []
    val_loss = []

    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    for epoch in range(epochs) :
        model.train()
        running_loss = 0.0

        for input , target in train_loader :
            input , target = input.to(device) , target.to(device)

            # 前向传播
            output = model(input)
            loss = loss_fn(output , target)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        # 验证集评估
        model.eval()
        with torch.no_grad() :
            val_output = model(X_val_tensor)
            epoch_val_loss = loss_fn(val_output , y_val_tensor)
            val_loss.append(epoch_val_loss.item())

        # 打印训练进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {epoch_val_loss.item():.6f}")

    return train_loss, val_loss

# 模型初始化
input_size = 7  # 7个特征
d_model = 32
nhead = 2
hidden_size = 16  # 增加隐藏单元数量
num_layers = 2    # 减少LSTM层数
output_size = 1   # 预测1个负荷值
dropout = 0.3     # 减少dropout

fusion_model = FusionModel(input_size , d_model , nhead , num_layers , output_size , dropout).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)



# 开始训练
epochs = 200
print(f"开始训练，共 {epochs} 个epoch...")
print("="*60)

train_losses, val_losses = train(
    model=fusion_model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_loader,
    X_val=X_val,
    y_val=y_val,
    epochs=epochs
)

print("="*60)
print("训练完成!")

# 最终测试集评估
fusion_model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    test_output = fusion_model(X_test_tensor)
    test_loss = loss_fn(test_output, y_test_tensor)

    print(f"测试集MSE Loss: {test_loss.item():.6f}")

    # 将标准化后的预测值和真实值转换回原始尺度
    test_pred_original = target_scaler.inverse_transform(test_output.cpu().numpy())
    test_true_original = target_scaler.inverse_transform(y_test_tensor.cpu().numpy())

    # 输出每个测试样本的预测值
    print("=" * 80)
    print("详细预测结果:")
    print("="*80)
    print(f"{'样本':<6} | {'预测值':<10} | {'真实值':<10} | {'误差':<10} | {'误差率':<8}")
    print("-"*80)

    for i in range(len(test_pred_original)):
        pred_val = test_pred_original[i][0]
        true_val = test_true_original[i][0]
        abs_error = abs(pred_val - true_val)
        error_rate = (abs_error / true_val) * 100 if true_val != 0 else 0

        print(f"{i+1:<6} | {pred_val:<10.2f} | {true_val:<10.2f} | {abs_error:<10.2f} | {error_rate:<8.1f}%")

    # 计算MAE和RMSE
    mae = np.mean(np.abs(test_pred_original - test_true_original))
    rmse = np.sqrt(np.mean((test_pred_original - test_true_original)**2))

    print("-"*80)
    print(f"测试集MAE: {mae:.2f}")
    print(f"测试集RMSE: {rmse:.2f}")

    # 添加一些统计信息
    print(f"\n预测统计:")
    print(f"预测值范围: {test_pred_original.min():.2f} ~ {test_pred_original.max():.2f}")
    print(f"真实值范围: {test_true_original.min():.2f} ~ {test_true_original.max():.2f}")
    print(f"平均误差率: {np.mean(np.abs((test_pred_original.flatten() - test_true_original.flatten()) / test_true_original.flatten())) * 100:.1f}%")
    print("="*80)

# 保存模型
torch.save({
    'model_state_dict': fusion_model.state_dict(),
    'scaler': scaler,
    'target_scaler': target_scaler,
    'model_config': {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'output_size': output_size,
        'dropout': dropout
    }
}, 'lstm_model.pth')

print("模型已保存为 'lstm_model.pth'")

