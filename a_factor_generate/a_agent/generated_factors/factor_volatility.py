import pandas as pd
import numpy as np

def factor_volatility(data, window=20):
    """
    计算波动率因子：最近window个交易日的收益率标准差
    
    参数:
    data: 包含股票价格数据的DataFrame
    window: 计算窗口，默认20个交易日
    
    返回:
    factor: 波动率因子值
    """
    # 计算日收益率
    returns = data['close'] / data['close'].shift(1) - 1
    
    # 计算波动率（标准差）
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # 年化
    
    return volatility