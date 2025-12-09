import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta


class PeopleFlow:
    """人流量模拟--正态分布"""

    def __init__(self, points, mu_idx, sigma_idx, peak_num, if_excel, q_each, if_load):
        self.points = points
        self.mu_idx = mu_idx
        self.sigma_idx = sigma_idx
        self.peak_num = peak_num
        self.if_excel = if_excel
        self.q_each = q_each
        self.if_load = if_load

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        x = np.arange(self.points)
        passengers = self.peak_num * np.exp(-(x - self.mu_idx) ** 2 / (2 * self.sigma_idx ** 2))
        passengers = passengers.astype(int)

        load = passengers * self.q_each
        if self.if_load:
            df = pd.DataFrame({
                "time": time_list, "passengers": passengers, "passengers_load": load
            })
        else:
            df = pd.DataFrame({
                "time": time_list, "passengers": passengers
            })
        return df

    def figure(self):
        df = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df["time"], df["passengers"], marker="+")
        plt.xlabel("Time")
        plt.ylabel("Passengers")
        plt.grid(True)
        plt.show()


class EquipHeap:
    """设备散热量"""

    def __init__(self, points, q_equip, if_excel=False):
        self.points = points
        self.q_equip = q_equip
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        equip_heap = np.full(self.points, self.q_equip)
        df_equip = pd.DataFrame({
            "time": time_list,
            "equip_heat": equip_heap
        })
        return df_equip

    def figure(self):
        df_equip = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df_equip["time"], df_equip["equip_heat"], marker="+")
        plt.xlabel("Time")
        plt.ylabel("Equip_heat (W)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class StructureLoad:
    """地铁结构热负荷模拟"""

    def __init__(self, points, q_structure, if_excel=False):
        self.points = points
        self.q_structure = q_structure
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        structure_load = np.full(self.points, self.q_structure)
        df_structure = pd.DataFrame({
            "time": time_list,
            "structure_load": structure_load
        })
        return df_structure

    def figure(self):
        df_structure = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df_structure["time"], df_structure["structure_load"], marker="+")
        plt.xlabel("Time")
        plt.ylabel("Structure Load (W)")
        plt.title("Subway Structure Heat Load Simulation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class VentilationLoad:
    """渗透风热负荷模拟--交替模式"""

    def __init__(self, points, q_vent, period=1, if_excel=False):
        self.points = points
        self.q_vent = q_vent
        self.period = period
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        # vent_load = np.array([(self.q_vent if (i // self.period) % 2 == 0 else 0) for i in range(self.points)])
        vent_load = []
        for i in range(self.points):
            if (i % 2):
                vent_load.append(0)
            else:
                vent_load.append(self.q_vent)

        df_vent = pd.DataFrame({
            "time": time_list,
            "vent_load": vent_load
        })
        return df_vent

    def figure(self):
        df_vent = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df_vent["time"], df_vent["vent_load"], marker="+")
        plt.xlabel("Time")
        plt.ylabel("Vent_load (W)")
        plt.title("Ventilation Heat Load Simulation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class TemperatureFlow:
    """温度模拟--正态分布（整数，无随机扰动）"""

    def __init__(self, points, mu_idx, sigma_idx, peak_temp, base_temp=25, if_excel=False):
        """
        points: 时间点数量
        mu_idx: 温度峰值索引
        sigma_idx: 峰值标准差
        peak_temp: 峰值温度
        base_temp: 基础温度
        if_excel: 是否导出 Excel
        """
        self.points = points
        self.mu_idx = mu_idx
        self.sigma_idx = sigma_idx
        self.peak_temp = peak_temp
        self.base_temp = base_temp
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        x = np.arange(self.points)

        # 正态分布公式生成温度
        temp = self.base_temp + (self.peak_temp - self.base_temp) * np.exp(
            -(x - self.mu_idx) ** 2 / (2 * self.sigma_idx ** 2))
        temp = temp.astype(int)  # 转为整数

        df = pd.DataFrame({
            "time": time_list,
            "temp": temp
        })

        if self.if_excel:
            df.to_excel("temperature.xlsx", index=False)
        return df

    def figure(self):
        df = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df["time"], df["temp"], marker="+")
        plt.xlabel("Time")
        plt.ylabel("Temperature (℃)")
        plt.grid(True)
        plt.show()


class HumFlow:
    """湿度模拟--正态分布（整数，无随机扰动）"""

    def __init__(self, points, mu_idx, sigma_idx, peak_hum, base_hum, if_excel=False):
        self.points = points
        self.mu_idx = mu_idx
        self.sigma_idx = sigma_idx
        self.peak_temp = peak_hum
        self.base_temp = base_hum
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        x = np.arange(self.points)

        # 正态分布公式生成温度
        hum = self.base_temp + (self.peak_temp - self.base_temp) * np.exp(
            -(x - self.mu_idx) ** 2 / (2 * self.sigma_idx ** 2))
        hum = hum.astype(int)  # 转为整数

        df = pd.DataFrame({
            "time": time_list,
            "hum": hum
        })

        if self.if_excel:
            df.to_excel("hum.xlsx", index=False)
        return df

    def figure(self):
        df = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df["time"], df["hum"], marker="+")
        plt.xlabel("Time")
        plt.ylabel("Hum (%)")
        plt.grid(True)
        plt.show()


class EquipNum:
    """冷水机组数量模拟--正态分布或峰值波动（整数）"""

    def __init__(self, points, mu_idx, sigma_idx, min_num=1, max_num=5, if_excel=False):
        self.points = points
        self.mu_idx = mu_idx
        self.sigma_idx = sigma_idx
        self.min_num = min_num
        self.max_num = max_num
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        x = np.arange(self.points)

        # 正态分布生成机组数量
        num = self.min_num + (self.max_num - self.min_num) * np.exp(-(x - self.mu_idx) ** 2 / (2 * self.sigma_idx ** 2))
        num = np.round(num).astype(int)

        df = pd.DataFrame({
            "time": time_list,
            "equip_num": num
        })

        if self.if_excel:
            df.to_excel("equip_num.xlsx", index=False)
        return df

    def figure(self):
        df = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df["time"], df["equip_num"], marker="+", color="green")
        plt.xlabel("Time")
        plt.ylabel("Chiller Number")
        plt.grid(True)
        plt.show()
