import pandas as pd

# 1. 读取数据
data = pd.read_csv('数据2.3.csv')

# （1）查看数据集中的缺失值
print("缺失值情况：")
print(data.isnull().sum())

# （2）用字符串'缺失数据'代替缺失值
data_str_fill = pd.read_csv('数据2.3.csv')
data_str_fill.fillna('缺失数据', inplace=True)
print("用字符串填充后的结果：")
print(data_str_fill)

# 用前后值填充缺失数据（向前填充）
data_ffill = pd.read_csv('数据2.3.csv')
data_ffill.fillna(method='ffill', inplace=True)
print("用前后值填充（向前填充）后的结果：")
print(data_ffill)

# 用变量均值填充缺失值
data_mean_fill = pd.read_csv('数据2.3.csv')
data_mean_fill.fillna(data_mean_fill.mean(), inplace=True)
print("用变量均值填充后的结果：")
print(data_mean_fill)

# 用变量中位数填充缺失值
data_median_fill = pd.read_csv('数据2.3.csv')
data_median_fill.fillna(data_median_fill.median(), inplace=True)
print("用变量中位数填充后的结果：")
print(data_median_fill)

# 用线性插值法填充缺失数据
data_interpolate_fill = pd.read_csv('数据2.3.csv')
data_interpolate_fill.interpolate(inplace=True)
print("用线性插值法填充后的结果：")
print(data_interpolate_fill)

# （3）重新读取数据集，删除数据集中的缺失值
data_dropna = pd.read_csv('数据2.3.csv')
data_dropna.dropna(inplace=True)
print("删除缺失值后的数据：")
print(data_dropna)

# （4）重新读取数据集，查看数据集中的重复值
data_check_duplicates = pd.read_csv('数据2.3.csv')
print("重复值情况：")
print(data_check_duplicates.duplicated().sum())

# （5）重新读取数据集，删除数据集中的重复值
data_drop_duplicates = pd.read_csv('数据2.3.csv')
data_drop_duplicates.drop_duplicates(inplace=True)
print("删除重复值后的数据：")
print(data_drop_duplicates)

# （6）重新读取数据集，删除变量列（V8、V9）、样本示例行（0、3、7）
data_drop_columns_and_rows = pd.read_csv('数据2.3.csv')
data_drop_columns_and_rows.drop(columns=['V8', 'V9'], inplace=True)
data_drop_columns_and_rows.drop(index=[0, 3, 7], inplace=True)
print("删除变量列和样本示例行后的数据：")
print(data_drop_columns_and_rows)

# （7）重新读取数据集，更改所有变量列名称为（var2、var3……），然后将 var2、var3 两列位置互换
data_rename_and_swap = pd.read_csv('数据2.3.csv')
new_columns = [f'var{i}' for i in range(2, data_rename_and_swap.shape[1] + 2)]
data_rename_and_swap.columns = new_columns
cols = list(data_rename_and_swap.columns)
cols[0], cols[1] = cols[1], cols[0]
data_rename_and_swap = data_rename_and_swap[cols]
print("更改列名并互换 var2 和 var3 后的数据：")
print(data_rename_and_swap)

# （8）改变 var2、var3 两列数据数据格式为字符串，并进行验证
data_str_format = data_rename_and_swap
data_str_format['var2'] = data_str_format['var2'].astype(str)
data_str_format['var3'] = data_str_format['var3'].astype(str)
print("var2 和 var3 转换为字符串后的数据类型验证：")
print(data_str_format.dtypes)

# （9）将 var5 的数据格式设置成百分比格式
data_percentage_format = data_str_format
data_percentage_format['var5'] = data_percentage_format['var5'].apply(lambda x: f'{x * 100}%')
print("var5 设置为百分比格式后的数据：")
print(data_percentage_format)