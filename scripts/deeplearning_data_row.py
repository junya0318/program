import os
import pandas as pd

input_dir = '/home/user_05/program/label_data/d_4ele_pilot_90_MATLAB'
output_dir = '/home/user_05/program/label_data/d_4ele_pilot_90_MATLAB_01'
cols = 995

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 100個のファイルを処理
for i in range(100):
    input_file = os.path.join(input_dir, f'd_50_{i}.csv')
    output_file = os.path.join(output_dir, f'd_50per_01_{i}.csv')

    if os.path.exists(input_file):
        d_10_data = pd.read_csv(input_file, header=None)

        # 行数を動的に決定
        rows = len(d_10_data)

        new_data = pd.DataFrame(0, index=range(rows), columns=range(cols))

        for index, row in d_10_data.iterrows():
            for col_index in row.dropna().astype(int):
                if col_index < cols:
                    new_data.at[index, col_index] = 1

        new_data.to_csv(output_file, index=False, header=False, chunksize=1000)
        print(f"{input_file} を変換し、{output_file} に保存しました。")
    else:
        print(f"ファイル {input_file} が存在しません。")
