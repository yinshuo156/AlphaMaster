import pandas as pd
import sys

file_path = r'c:\Users\Administrator\Desktop\alpha-master\data\a_share\csi500data\daily_data.csv'

try:
    # 读取CSV文件的前几行
    df = pd.read_csv(file_path, nrows=5)
    
    # 打印列名
    print("CSV文件列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. '{col}'")
    
    # 打印前几行数据
    print("\n前5行数据:")
    print(df.head())
    
    # 打印数据信息
    print("\n数据形状:", df.shape)
    
    sys.exit(0)
except Exception as e:
    print(f"读取CSV文件失败: {str(e)}")
    sys.exit(1)