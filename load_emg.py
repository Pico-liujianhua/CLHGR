import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import generate_mask

def load_EMG_from_csv(dataset, dataset_name='emg', load_tp=True):
    """
    直接从已处理的CSV文件加载EMG数据，转换为类似UCR的格式
    
    参数:
        data_root (str): 包含emg_train.csv和emg_test.csv的目录
        dataset_name (str): 数据集名称标识
        load_tp (bool): 是否加载时间点信息
        
    返回:
        tuple: (train_data, train_labels, test_data, test_labels)
               train_data/test_data是包含'x'和'mask'的字典
    """
    # 加载CSV文件

    train_path = os.path.join('./datasets/emg', dataset + "_train.csv")
    test_path = os.path.join('./datasets/emg', dataset + "_test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 提取数据和标签
    train_labels = train_df['label'].values.astype(int)
    test_labels = test_df['label'].values.astype(int)
    
    train = train_df.drop(columns=['label']).values.astype(np.float64) #(n_samples, seq_len）
    test = test_df.drop(columns=['label']).values.astype(np.float64)
    
    # 调整维度为NTC (样本数×时间步×通道数)
    train = train[..., np.newaxis]  # shape: (n_samples, seq_len, 1)
    test = test[..., np.newaxis]
    
    # 生成全1的mask (假设没有缺失值)
    mask_tr = generate_mask(train)
    mask_te = generate_mask(test)
    
    # 数据标准化 (全局标准化)
    scaler = StandardScaler()
    train_shape = train.shape
    test_shape = test.shape
    
    # 展平后标准化再恢复形状
    train = scaler.fit_transform(train.reshape(-1, 1)).reshape(train_shape)
    test = scaler.transform(test.reshape(-1, 1)).reshape(test_shape)
    
    # 添加时间点信息 (可选)
    if load_tp:
        seq_len = train.shape[1]
        tp = np.linspace(0, 1, seq_len, endpoint=True).reshape(1, -1, 1)
        
        train = np.concatenate((train, np.repeat(tp, train.shape[0], axis=0)), axis=-1)
        test = np.concatenate((test, np.repeat(tp, test.shape[0], axis=0)), axis=-1)
    
    # 返回与UCR相同的结构
    return (
        {'x': train, 'mask': mask_tr},  # 训练数据
        train_labels,                   # 训练标签
        {'x': test, 'mask': mask_te},   # 测试数据
        test_labels                      # 测试标签
    )