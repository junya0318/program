import os
import pandas as pd

input_dir = '/home/user_05/program/label_data/d_top_4ele_0.1lam_sna10000_MATLAB' # 元ファイルが保存されているディレクトリ
output_dir = '/home/user_05/program/label_data/d_top_4ele_0.1lam_sna10000_MATLAB_01'  # 変換後のファイルを保存するディレクトリ

# 列数と行数の指定
rows = 30000
cols = 995

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 100個のファイルを処理するループ
for i in range(100):
    # 入力ファイルのパス
    input_file = os.path.join(input_dir, f'd_1_{i}.csv')
    
    # 出力ファイルのパス
    output_file = os.path.join(output_dir, f'd_1per_01_{i}.csv')
    
    # 入力ファイルを読み込み
    if os.path.exists(input_file):
        d_10_data = pd.read_csv(input_file, header=None)
        
        # すべて0で初期化されたデータフレームを作成
        new_data = pd.DataFrame(0, index=range(rows), columns=range(cols))
        
        # d_10_data の値に基づいて、対応する列に1を設定
        for index, row in d_10_data.iterrows():
            if index < rows:
                for col_index in row.dropna().astype(int):
                    if col_index < cols:
                        new_data.at[index, col_index] = 1
        
        # 変換したデータをCSVファイルとして保存
        new_data.to_csv(output_file, index=False, header=False, chunksize=1000)
        
        print(f"{input_file} を変換し、{output_file} に保存しました。")
    else:
        print(f"ファイル {input_file} が存在しません。")
