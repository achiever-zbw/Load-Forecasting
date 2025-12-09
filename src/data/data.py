import pandas as pd
from utils.classes import *

people_flow = PeopleFlow(points=288, mu_idx=144, sigma_idx=50, peak_num=2000, if_excel=False , q_each = 0.182 , if_load = False)
# people_flow.figure()
temp = TemperatureFlow(points = 288 , mu_idx = 144 , sigma_idx = 36 , peak_temp = 32 , base_temp = 25 , if_excel = False)
hum = HumFlow(points = 288 , mu_idx = 144 , sigma_idx = 36 , peak_hum = 70 , base_hum = 45)
equip_heap = EquipHeap(points=288, q_equip=500, if_excel=False)
structure_load = StructureLoad(points=288, q_structure=300, if_excel=False)
ventilation_load = VentilationLoad(points=288, q_vent=1, period=2, if_excel=False)
equip_num = EquipNum(points=288 , mu_idx=144 , sigma_idx=36 , min_num=1 , max_num=2 )
# hum.figure()

# ventilation_load.figure()
# 生成四个数据框
df_people_flow = people_flow.make()
df_equip_heap = equip_heap.make()
df_structure_load = structure_load.make()
df_ventilation_load = ventilation_load.make()
df_temp = temp.make()
df_hum = hum.make()  # 湿度
df_equip_num = equip_num.make()
# temp.figure()
# 美化 time 列，保留时间的分钟表示，并转换为 HH:MM 格式
for df in (df_people_flow, df_equip_heap, df_structure_load, df_hum ,  df_ventilation_load , df_temp , df_equip_num):
    # 转换为分钟数并格式化为 HH:MM
    df["time"] = (df["time"].dt.total_seconds() / 60).astype(int)
    df["time"] = pd.to_datetime(df["time"], unit="m").dt.strftime("%H:%M")


# 重新合并四个数据框
df_total = (
    df_people_flow                         # 人流量
    .merge(df_ventilation_load, on="time") # 渗透风（0 1）
    .merge(df_structure_load, on="time")   # 深埋（结构）
    .merge(df_temp, on="time")             # 温度
    .merge(df_hum , on = "time")           # 湿度
    # .merge(df_equip_heap, on="time")       # 设备热
    .merge(df_equip_num , on = "time")     # 冷水机组
)

# 保证列顺序
df_total = df_total[["time" ,"passengers","structure_load" , "vent_load" , "temp" , "hum" , "equip_num"]]



# 导出到一个Excel工作表
df_total.to_excel("数据总表.xlsx", index=False)

print("数据已导出到 '数据总表.xlsx'")


