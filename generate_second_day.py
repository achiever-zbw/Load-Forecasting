import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_second_day_data():
    """基于第一天数据生成第二天的数据，添加合理的变化"""

    # 读取原始数据
    input_data = pd.read_excel('data/raw/数据总表.xlsx')
    target_data = pd.read_excel('data/raw/负荷数据总表.xlsx')

    # 复制第一天数据作为第二天的基础
    day2_input = input_data.copy()
    day2_target = target_data.copy()

    # 添加一些合理的随机变化来模拟第二天的情况
    np.random.seed(42)  # 确保可重现

    # 1. 对乘客流量添加变化 (±15%)
    passenger_variation = np.random.normal(1.0, 0.1, len(day2_input))
    day2_input['passengers'] = np.clip(
        day2_input['passengers'] * passenger_variation,
        day2_input['passengers'] * 0.7,
        day2_input['passengers'] * 1.3
    ).astype(int)

    # 2. 对温度添加小幅变化 (±1度)
    temp_variation = np.random.normal(0, 0.5, len(day2_input))
    day2_input['temp'] = np.clip(
        day2_input['temp'] + temp_variation,
        24, 33
    ).round(1)

    # 3. 对湿度添加小幅变化 (±3%)
    hum_variation = np.random.normal(0, 1.5, len(day2_input))
    day2_input['hum'] = np.clip(
        day2_input['hum'] + hum_variation,
        42, 73
    ).round().astype(int)

    # 4. 随机调整设备数量 (偶尔有设备维护)
    equip_change_prob = 0.05  # 5%概率发生变化
    for i in range(len(day2_input)):
        if np.random.random() < equip_change_prob:
            day2_input.loc[i, 'equip_num'] = 3 - day2_input.loc[i, 'equip_num']  # 1变2，2变1

    # 5. 时间保持不变 (还是00:00到23:55)
    # 时间列保持原样，因为我们只关心时间模式

    # 6. 对负荷数据添加相应变化
    # 负荷变化主要基于乘客和温度的变化
    load_variation = np.random.normal(1.0, 0.08, len(day2_target))  # ±8%变化

    # 计算乘客负荷变化
    passengers_load_change = (day2_input['passengers'] - input_data['passengers']) * 0.2

    # 调整乘客负荷
    day2_target['passengers_load'] = day2_target['passengers_load'] * load_variation + passengers_load_change

    # 确保负荷为正值
    day2_target['passengers_load'] = np.maximum(day2_target['passengers_load'], 1.0)

    # 重新计算总负荷
    day2_target['total_load'] = day2_target['passengers_load'] + day2_target['structure_load'] + day2_target['vent_load']

    print("第二天数据生成完成!")
    print(f"输入数据形状: {day2_input.shape}")
    print(f"负荷数据形状: {day2_target.shape}")
    print()
    print("第二天数据样本:")
    print("输入数据前5行:")
    print(day2_input.head())
    print("\n负荷数据前5行:")
    print(day2_target.head())

    # 合并两天数据
    combined_input = pd.concat([input_data, day2_input], ignore_index=True)
    combined_target = pd.concat([target_data, day2_target], ignore_index=True)

    print(f"\n合并后的数据形状:")
    print(f"输入数据: {combined_input.shape}")
    print(f"负荷数据: {combined_target.shape}")

    return combined_input, combined_target

def save_combined_data():
    """保存合并后的数据到新的Excel文件"""

    combined_input, combined_target = generate_second_day_data()

    # 保存备份原文件
    import shutil
    from datetime import datetime

    backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 备份原始文件
    shutil.copy('data/raw/数据总表.xlsx', f'data/raw/数据总表_backup_{backup_suffix}.xlsx')
    shutil.copy('data/raw/负荷数据总表.xlsx', f'data/raw/负荷数据总表_backup_{backup_suffix}.xlsx')

    # 保存合并后的数据
    combined_input.to_excel('data/raw/数据总表.xlsx', index=False)
    combined_target.to_excel('data/raw/负荷数据总表.xlsx', index=False)

    print(f"\n✅ 数据已成功保存!")
    print(f"📁 原始文件已备份为: 数据总表_backup_{backup_suffix}.xlsx")
    print(f"📁 原始文件已备份为: 负荷数据总表_backup_{backup_suffix}.xlsx")
    print(f"💾 新数据已保存到: 数据总表.xlsx 和 负荷数据总表.xlsx")

    # 显示一些统计信息
    print(f"\n📊 数据统计:")
    print(f"总数据点数: {len(combined_input)} (原: 288, 新: {len(combined_input)})")
    print(f"乘客流量范围: {combined_input['passengers'].min()} ~ {combined_input['passengers'].max()}")
    print(f"温度范围: {combined_input['temp'].min()} ~ {combined_input['temp'].max()}°C")
    print(f"负荷范围: {combined_target['total_load'].min():.1f} ~ {combined_target['total_load'].max():.1f}")

if __name__ == "__main__":
    save_combined_data()