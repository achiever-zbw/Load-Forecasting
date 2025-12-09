import pandas as pd
from datetime import datetime
import sys
import os
from src.utils.classes import *

def generate_one_day_data():
    """生成一天的地铁空调负荷数据"""

    # 使用现有的类生成数据，参数与data.py保持一致
    people_flow = PeopleFlow(points=288, mu_idx=144, sigma_idx=50, peak_num=2000,
                            if_excel=False, q_each=0.182, if_load=False)
    temp = TemperatureFlow(points=288, mu_idx=144, sigma_idx=36, peak_temp=32, base_temp=25, if_excel=False)
    hum = HumFlow(points=288, mu_idx=144, sigma_idx=36, peak_hum=70, base_hum=45)
    structure_load = StructureLoad(points=288, q_structure=300, if_excel=False)
    ventilation_load = VentilationLoad(points=288, q_vent=1, period=2, if_excel=False)
    equip_num = EquipNum(points=288, mu_idx=144, sigma_idx=36, min_num=1, max_num=2)

    # 生成各个数据
    df_people_flow = people_flow.make()              # 人流量
    df_temp = temp.make()                            # 温度
    df_hum = hum.make()                              # 湿度
    df_structure_load = structure_load.make()        # 结构
    df_ventilation_load = ventilation_load.make()    # 渗透风
    df_equip_num = equip_num.make()                  # 设备数

    # 格式化时间列为HH:MM格式
    for df in (df_people_flow, df_temp, df_hum, df_structure_load, df_ventilation_load, df_equip_num):
        df["time"] = (df["time"].dt.total_seconds() / 60).astype(int)
        df["time"] = pd.to_datetime(df["time"], unit="m").dt.strftime("%H:%M")

    # 合并数据框
    df_total = (
        df_people_flow
        .merge(df_ventilation_load, on="time")
        .merge(df_structure_load, on="time")
        .merge(df_temp, on="time")
        .merge(df_hum, on="time")
        .merge(df_equip_num, on="time")
    )

    # 调整列顺序
    df_total = df_total[["time", "passengers", "structure_load", "vent_load", "temp", "hum", "equip_num"]]

    return df_total

