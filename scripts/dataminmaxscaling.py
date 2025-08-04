import pandas as pd
import os

# 入力と出力ディレクトリの指定
input_dir = '/home/user_05/program/R_d50per_random_90'  # 元ファイルが保存されているディレクトリ
output_dir = '/home/user_05/program/R_d50per_random_90_minmax_scaled'  # 変換後のファイルを保存するディレクトリ

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# 100個のファイルを処理するループ
for i in range(100):
    # 入力ファイルのパス
    input_file = os.path.join(input_dir, f'R_50_{i}.csv')
    
    # 出力ファイルのパス
    output_file = os.path.join(output_dir, f'R_50_{i}.csv')
    
    # 入力ファイルを読み込み、Min-Maxスケーリング
    if os.path.exists(input_file):
        # CSVをDataFrameに読み込み
        data = pd.read_csv(input_file, header=None)
        
        # Min-Maxスケーリング
        min_values = data.min()
        max_values = data.max()
        scaled_data = (data - min_values) / (max_values - min_values)
        
        # 変換したデータをCSVファイルとして保存
        scaled_data.to_csv(output_file, index=False, header=False)
        
        print(f"{input_file} をMin-Maxスケーリングし、{output_file} に保存しました。")
    else:
        print(f"ファイル {input_file} が存在しません。")
