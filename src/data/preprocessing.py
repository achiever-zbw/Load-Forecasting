import numpy as np

def create_sequences(features , target , time_steps) :
    """从特征和目标序列中创建时间序列样本"""
    X , y = [] , []
    for i in range(len(features) - time_steps) :
        X.append(features[i : i + time_steps])
        y.append(target[i + time_steps])

    return np.array(X) , np.array(y)
