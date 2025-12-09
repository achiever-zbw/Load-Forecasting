import pandas as pd
import numpy as np

def load_and_process_data(file_path, time_step=10):
    """
    读取并处理四种负荷数据，构建时间序列样本
    """

    # 读取Excel
    df = pd.read_excel(file_path)

    # 确认包含所需列
    required_cols = ["time", "passengers", "passengers_load", "vent_load", "equip_heat", "structure_load"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    features = ["passengers_load", "vent_load", "equip_heat", "structure_load"]
    X = df[features].values
    y = df[["passengers_load", "equip_heat", "structure_load"]].sum(axis=1).values


    # 构造时间序列
    def create_sequences(X, y, time_step):
        Xs, ys = [], []
        for i in range(len(X) - time_step):
            Xs.append(X[i:(i + time_step)])
            ys.append(y[i + time_step])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_sequences(X, y, time_step)

    print("X_seq shape:", X_seq.shape)  # (样本数, time_step, 特征数)
    print("y_seq shape:", y_seq.shape)  # (样本数,)

    return X_seq, y_seq


if __name__ == "__main__":
    file = "data/raw/四种数据总表.xlsx"

    # 方式A: vent_load作为0/1特征
    X_A, y_A = load_and_process_data(file, time_step=10)
    print(X_A)

