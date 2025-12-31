import os
import csv

def load_csv_data(filename, exclude_kv=False):
    """
    读取CSV文件并将数据转换为指定的列表格式。
    
    Args:
        filename (str): CSV文件名
        exclude_kv (bool): 是否排除包含 'KV' 的列 (用于 bwd 数据)
        
    Returns:
        providers (list): 处理后的表头列表
        times_data (list): 包含 (Method, [values...]) 元组的列表
    """
    providers = []
    times_data = []

    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found.")
        return [], []

    with open(filename, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # 1. 处理表头 (Header)
        try:
            headers = next(reader)
        except StopIteration:
            return [], [] # 空文件

        # 确定我们需要读取哪些列的索引
        # 索引 0 是 "Method"，我们跳过它作为数据列，但用它作为行名
        target_indices = []
        
        for idx, col_name in enumerate(headers):
            if idx == 0:
                continue # 跳过 Method 列
            
            # 如果是 bwd 文件，根据你的示例，通常不包含 KV 数据
            if exclude_kv and "KV" in col_name:
                continue
            
            # 记录需要读取的列索引
            target_indices.append(idx)
            
            # 格式化 Provider 字符串: "BS1 S2048" -> "BS1\nS2048"
            formatted_name = col_name.replace(" ", "\n")
            providers.append(formatted_name)

        # 2. 处理数据行 (Rows)
        for row in reader:
            if not row: continue # 跳过空行
            
            method_name = row[0]
            row_values = []
            
            for idx in target_indices:
                val_str = row[idx] if idx < len(row) else ""
                try:
                    # 尝试转换为浮点数
                    val = float(val_str)
                except ValueError:
                    # 如果转换失败（例如空字符串），设为0或者处理异常
                    val = 0.0 
                row_values.append(val)
            
            times_data.append((method_name, row_values))

    return providers, times_data

