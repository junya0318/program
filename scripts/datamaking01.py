import os
import pandas as pd


# CSVファイルの読み込み
d_10_1 = pd.read_csv('/home/user_05/program/d_10per_75/d_10_1.csv', header=None)

# 行数と列数
rows = 30000
cols = 995

# 全て0で初期化されたデータフレームを作成
test_data = pd.DataFrame(0, index=range(rows), columns=range(cols))

# d_10_1 の値に基づいて、対応する列に1を設定
for index, row in d_10_1.iterrows():
    if index < rows:
        for col_index in row.dropna().astype(int):
            if col_index < cols:
                test_data.at[index, col_index] = 1

# new_data をCSVファイルに保存
output_path = '/home/user_05/program/d_10per_75_01/test_data.csv'
test_data.to_csv(output_path, index=False, header=False, chunksize=1000)

print(test_data.head())
print(f"新しいデータを {output_path} として保存しました。")

