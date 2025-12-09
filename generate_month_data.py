import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar

class MonthDataGenerator:
    def __init__(self, start_date="2024-06-01"):
        """
        初始化月数据生成器

        Args:
            start_date: 起始日期，格式 YYYY-MM-DD，默认6月1日(夏季)
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.base_data = None
        self.load_base_data()

    def load_base_data(self):
        """加载基础数据(第一天)"""
        self.base_data = pd.read_excel('data/raw/数据总表_backup_20251209_192428.xlsx')
        self.base_target = pd.read_excel('data/raw/负荷数据总表_backup_20251209_192428.xlsx')

    def is_weekend(self, date):
        """判断是否为周末"""
        return date.weekday() >= 5  # 周六=5, 周日=6

    def is_holiday(self, date):
        """判断是否为节假日(简化版)"""
        # 这里可以添加中国的节假日逻辑
        # 暂时返回False
        return False

    def get_day_type_factor(self, date):
        """获取日期类型的影响因子"""
        if self.is_weekend(date):
            return {
                'passenger_factor': 0.7,  # 周末乘客减少
                'temp_offset': 1.0,       # 温度基本不变
                'load_base_factor': 0.85  # 基础负荷略低
            }
        elif self.is_holiday(date):
            return {
                'passenger_factor': 0.6,  # 节假日乘客更少
                'temp_offset': 1.0,
                'load_base_factor': 0.8
            }
        else:
            return {
                'passenger_factor': 1.0,  # 工作日正常
                'temp_offset': 1.0,
                'load_base_factor': 1.0
            }

    def get_hour_pattern(self, hour, day_type_factor):
        """获取小时模式调整因子"""
        # 模拟早晚高峰模式
        if 7 <= hour <= 9:  # 早高峰
            return 1.3
        elif 17 <= hour <= 19:  # 晚高峰
            return 1.4
        elif 10 <= hour <= 16:  # 白天
            return 1.1
        elif 22 <= hour or hour <= 5:  # 深夜到凌晨
            return 0.6
        else:  # 其他时间
            return 1.0

    def get_seasonal_temp(self, date, base_temp):
        """获取季节性温度变化"""
        # 模拟夏季6月的温度变化
        day_of_year = date.timetuple().tm_yday

        # 温度在小范围内波动，模拟夏季特征
        seasonal_variation = 2 * np.sin(2 * np.pi * day_of_year / 30)  # 30天周期
        daily_variation = np.sin(2 * np.pi * day_of_year / 365) * 0.5  # 年度变化

        return base_temp + seasonal_variation + daily_variation

    def generate_day_data(self, day_offset):
        """生成指定天的数据"""
        current_date = self.start_date + timedelta(days=day_offset)
        day_type_factor = self.get_day_type_factor(current_date)

        # 复制基础数据结构
        day_input = self.base_data.copy()
        day_target = self.base_target.copy()

        # 设置随机种子，确保同一天的变化可重现
        np.random.seed(42 + day_offset)

        # 对每个时间点进行调整
        for i, row in day_input.iterrows():
            # 获取当前时间的小时
            time_str = row['time']
            hour = int(time_str.split(':')[0])

            # 计算小时模式因子
            hour_factor = self.get_hour_pattern(hour, day_type_factor)

            # 1. 调整乘客流量
            passenger_variation = np.random.normal(1.0, 0.15)  # ±15%随机变化
            day_input.loc[i, 'passengers'] = np.clip(
                day_input.loc[i, 'passengers'] *
                day_type_factor['passenger_factor'] *
                hour_factor *
                passenger_variation,
                10, 3000  # 合理范围
            ).astype(int)

            # 2. 调整温度(考虑季节性和小时)
            base_temp = day_input.loc[i, 'temp']
            seasonal_temp = self.get_seasonal_temp(current_date, base_temp)
            hourly_temp_variation = np.sin(2 * np.pi * hour / 24) * 2  # 小时温度变化
            random_temp_variation = np.random.normal(0, 0.3)

            final_temp = seasonal_temp + hourly_temp_variation + random_temp_variation
            day_input.loc[i, 'temp'] = np.clip(final_temp, 20, 38).round(1)

            # 3. 调整湿度
            hum_variation = np.random.normal(0, 2)
            day_input.loc[i, 'hum'] = np.clip(
                day_input.loc[i, 'hum'] + hum_variation,
                30, 80
            ).round().astype(int)

            # 4. 偶尔调整设备数量
            if np.random.random() < 0.08:  # 8%概率
                day_input.loc[i, 'equip_num'] = min(3, day_input.loc[i, 'equip_num'] + 1)

        # 调整负荷数据
        load_base_variation = np.random.normal(1.0, 0.1, len(day_target))

        # 计算乘客负荷变化
        passenger_ratio = day_input['passengers'] / self.base_data['passengers']

        # 调整乘客负荷
        day_target['passengers_load'] = (
            day_target['passengers_load'] *
            load_base_variation *
            passenger_ratio *
            day_type_factor['load_base_factor']
        )

        # 确保负荷为正值
        day_target['passengers_load'] = np.maximum(day_target['passengers_load'], 0.5)

        # 重新计算总负荷
        day_target['total_load'] = (
            day_target['passengers_load'] +
            day_target['structure_load'] +
            day_target['vent_load']
        )

        # 添加日期信息(虽然不用于训练，但有助于分析)
        day_input['date'] = current_date.strftime('%Y-%m-%d')
        day_input['day_of_week'] = current_date.weekday()
        day_input['is_weekend'] = int(self.is_weekend(current_date))

        return day_input, day_target, current_date

    def generate_month_data(self, num_days=30):
        """生成一个月的数据"""
        print(f"🔄 开始生成 {num_days} 天的数据...")

        all_inputs = []
        all_targets = []
        all_dates = []

        for day in range(num_days):
            current_date = self.start_date + timedelta(days=day)
            print(f"📅 生成第 {day+1}/{num_days} 天: {current_date.strftime('%Y-%m-%d %A')}", end="")

            day_input, day_target, date = self.generate_day_data(day)
            all_inputs.append(day_input)
            all_targets.append(day_target)
            all_dates.append(date)

            # 显示统计信息
            avg_passengers = day_input['passengers'].mean()
            avg_temp = day_input['temp'].mean()
            avg_load = day_target['total_load'].mean()

            print(f" | 乘客: {avg_passengers:.0f} | 温度: {avg_temp:.1f}°C | 负荷: {avg_load:.0f}")

        # 合并所有数据
        combined_input = pd.concat(all_inputs, ignore_index=True)
        combined_target = pd.concat(all_targets, ignore_index=True)

        print(f"✅ 数据生成完成!")
        print(f"📊 总数据点数: {len(combined_input)} (原: 288, 新: {len(combined_input)})")

        return combined_input, combined_target

    def save_month_data(self, num_days=30):
        """保存一个月的数据"""
        # 创建备份
        import shutil
        backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

        shutil.copy('data/raw/数据总表.xlsx', f'data/raw/数据总表_backup_before_month_{backup_suffix}.xlsx')
        shutil.copy('data/raw/负荷数据总表.xlsx', f'data/raw/负荷数据总表_backup_before_month_{backup_suffix}.xlsx')

        # 生成数据
        combined_input, combined_target = self.generate_month_data(num_days)

        # 保存数据
        combined_input.to_excel('data/raw/数据总表.xlsx', index=False)
        combined_target.to_excel('data/raw/负荷数据总表.xlsx', index=False)

        print(f"💾 数据已保存!")
        print(f"📁 备份文件: 数据总表_backup_before_month_{backup_suffix}.xlsx")

        # 显示详细统计
        print(f"\n📈 月份数据统计:")
        print(f"总天数: {num_days}")
        print(f"总数据点: {len(combined_input)}")
        print(f"乘客流量范围: {combined_input['passengers'].min()} ~ {combined_input['passengers'].max()}")
        print(f"温度范围: {combined_input['temp'].min():.1f} ~ {combined_input['temp'].max():.1f}°C")
        print(f"湿度范围: {combined_input['hum'].min()} ~ {combined_input['hum'].max()}%")
        print(f"负荷范围: {combined_target['total_load'].min():.1f} ~ {combined_target['total_load'].max():.1f}")

        # 工作日vs周末统计
        if 'is_weekend' in combined_input.columns:
            weekend_data = combined_input[combined_input['is_weekend'] == 1]
            weekday_data = combined_input[combined_input['is_weekend'] == 0]

            print(f"\n📊 工作日 vs 周末对比:")
            print(f"工作日平均乘客: {weekday_data['passengers'].mean():.0f}")
            print(f"周末平均乘客: {weekend_data['passengers'].mean():.0f}")
            print(f"工作日平均负荷: {combined_target[combined_input['is_weekend'] == 0]['total_load'].mean():.1f}")
            print(f"周末平均负荷: {combined_target[combined_input['is_weekend'] == 1]['total_load'].mean():.1f}")

def main():
    """主函数"""
    print("🚀 开始生成一个月的地铁空调负荷数据...")
    print("=" * 60)

    # 创建数据生成器(从6月1日开始，夏季)
    generator = MonthDataGenerator(start_date="2024-06-01")

    # 生成30天的数据
    generator.save_month_data(num_days=30)

    print("=" * 60)
    print("🎉 一个月数据生成完成! 现在可以开始训练模型了!")

if __name__ == "__main__":
    main()