import pandas as pd
import numpy as np

def factor_volume_change(data, short_window=5, long_window=20):
    """
    计算成交量变化率因子：短期平均成交量与长期平均成交量的比率
    
    参数:
    data: 包含成交量数据的DataFrame
    short_window: 短期窗口，默认5个交易日
    long_window: 长期窗口，默认20个交易日
    
    返回:
    factor: 成交量变化率因子值
    """
    # 计算短期平均成交量
    short_vol_avg = data['volume'].rolling(window=short_window).mean()
    
    # 计算长期平均成交量
    long_vol_avg = data['volume'].rolling(window=long_window).mean()
    
    # 计算比率作为因子值
    volume_ratio = short_vol_avg / long_vol_avg
    
    return volume_ratio