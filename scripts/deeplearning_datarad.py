import os
import pandas as pd
import numpy as np

# 入出力ディレクトリ
input_dir = '/home/user_05/program/input_data/desired_4ele_pilot_50per_theta_MATLAB'
output_dir = '/home/user_05/program/input_data/desired_4ele_pilot_50per_theta_MATLAB_rad'

# 出力ディレクトリがなければ作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 100個のファイルを処理（DoA_50_0.csv ～ DoA_50_99.csv）
for i in range(100):
    input_file = os.path.join(input_dir, f'desired_50_{i}.csv')
    output_file = os.path.join(output_dir, f'desired_50rad_{i}.csv')

    if os.path.exists(input_file):
        doa_deg = pd.read_csv(input_file, header=None)

        # ラジアンに変換
        doa_rad = np.radians(doa_deg.values)
        doa_rad_df = pd.DataFrame(doa_rad)

        # 出力（ヘッダー・インデックスなし）
        doa_rad_df.to_csv(output_file, index=False, header=False)
        print(f"{input_file} を変換して {output_file} に保存しました。")
    else:
        print(f"ファイル {input_file} が存在しません。")
