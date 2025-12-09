import pandas as pd

class LoadCalculator:
    """负荷计算"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_excel(file_path)
        # 检查必要的列是否存在
        required_cols = ["time", "passengers", "structure_load", "vent_load"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"数据表中缺少必要列: {col}")

    def calculate_passengers_load(self):
        """计算人流量负荷 (W)"""
        # 每个人的负荷为0.182 KW
        return self.df["passengers"] * 0.182

    def calculate_structure_load(self):
        """计算结构负荷 (KW)"""
        return self.df["structure_load"]

    def calculate_vent_load(self):
        """计算 渗透风负荷 """
        return self.df["vent_load"] * 50

    def calculate_total_load(self):
        """计算每个时间点的总负荷 (KW)"""
        passengers_load = self.calculate_passengers_load()
        # 总负荷 = 人流量负荷 + 结构负荷 + 渗透风负荷
        total_load = passengers_load + self.calculate_structure_load() + self.calculate_vent_load()
        return total_load

    def get_timepoint_loads(self, time_index=None):

        # 返回所有时间点的数据
        result_df = pd.DataFrame({
            "time": self.df["time"],
            "passengers_load": self.calculate_passengers_load(),
            "structure_load": self.calculate_structure_load(),
            "vent_load": self.calculate_vent_load(),
            "total_load": self.calculate_total_load()
        })
        return result_df

    def export_to_excel(self, output_file):
        """
        将所有时间点的负荷数据导出到Excel文件
        """
        # 获取所有时间点的负荷数据
        load_data = self.get_timepoint_loads()

        # 导出到Excel
        load_data.to_excel(output_file, index=False)
        print(f"负荷数据已导出到: {output_file}")
        print(f"数据包含 {len(load_data)} 个时间点")
        print(f"列名: {list(load_data.columns)}")

        return output_file


def __main__():
    file_path = "data/raw/数据总表.xlsx"
    calculator = LoadCalculator(file_path)
    output_file = calculator.export_to_excel("data/raw/一个月负荷数据总表.xlsx")

    calculator.export_to_excel(output_file)
    print(f"所有时间点的负荷数据已保存到: {output_file}")


if __name__ == "__main__":
    __main__()