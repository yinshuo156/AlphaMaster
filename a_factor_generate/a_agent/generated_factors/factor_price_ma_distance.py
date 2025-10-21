import pandas as pd
import numpy as np

def factor_price_ma_distance(data, window=20):
    """
    计算价格与均线距离因子：价格与移动平均线的偏离程度
    
    参数:
    data: 包含股票价格数据的DataFrame
    window: 移动平均线窗口，默认20个交易日
    
    返回:
    factor: 价格均线距离因子值
    """
    # 计算移动平均线
    ma = data['close'].rolling(window=window).mean()
    
    # 计算偏离程度
    distance = (data['close'] - ma) / ma
    
    return distance