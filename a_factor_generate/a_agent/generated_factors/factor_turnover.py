import pandas as pd
import numpy as np

def factor_turnover(data, window=20):
    """
    计算换手率因子：成交量与流通市值的比率
    
    参数:
    data: 包含成交量和流通市值数据的DataFrame
    window: 计算窗口，默认20个交易日
    
    返回:
    factor: 换手率因子值
    """
    # 计算换手率
    turnover = data['volume'] / data['circ_mv']
    
    # 计算平均换手率
    avg_turnover = turnover.rolling(window=window).mean()
    
    return avg_turnover