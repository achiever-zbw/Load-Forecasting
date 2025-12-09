import pandas as pd
from datetime import datetime, timedelta
from src.utils.one_day_data import generate_one_day_data

# 生成一段时间的所有数据，一个月
def get_all_data(start_date="2025-06-01", days=30):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    all_data = []

    print("=" * 50)

    # 开始生成每一天的数据
    for day in range(days):
        current_date = start_dt + timedelta(days=day)
        date_str = current_date.strftime("%Y-%m-%d")
        # 生成单日数据
        daily_data = generate_one_day_data()

        # 添加到全月数据列表
        all_data.append(daily_data)

    # 合并所有天的数据
    print(f"合并 {days} 天的数据")
    month_data = pd.concat(all_data, ignore_index=True)

    # 统一列顺序
    columns_order = [
        'time', 'passengers',
        'structure_load', 'vent_load',
        'temp', 'hum', 'equip_num'
    ]
    month_data = month_data[columns_order]

    return month_data


def main():

    # 生成整个月的数据
    month_data = get_all_data(start_date="2025-06-01", days=30)

    print(f"\n数据生成完成")
    print(f"总记录数: {len(month_data)}")
    print(f"列数: {len(month_data.columns)}")
    print(f"列名: {list(month_data.columns)}")

    # 保存到 Excel
    output_filename = "data/raw/一个月数据总表.xlsx"

    month_data.to_excel(output_filename, index=False)
    print(f"\n文件已成功保存为: {output_filename}")


if __name__ == "__main__":
    main()
