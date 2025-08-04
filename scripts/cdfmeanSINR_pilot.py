"""
accuracy making myself

"""
import tensorflow as tf
import math
from tqdm import tqdm
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
import time
from scipy.linalg import svd, pinv
from scipy.signal import find_peaks
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


# 使用可能なGPUを取得
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # 特定のGPUを指定 (例: GPU 1を使用)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0].name}")
        
        # メモリの成長を許可（推奨設定）
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found.")


class MLP(tf.keras.Model):
    '''多層パーセプトロン
    
    Attributes:
      l1(Dense): 隠れ層
      l2(Dense): 出力層
    '''
    def __init__(self, hidden_dim,hidden_dim2, hidden_dim3, hidden_dim4,output_dim):# # ,hidden_dim5,hidden_dim6
        '''
        Parameters:
          hidden_dim(int): 隠れ層のユニット数(次元)
          output_dim(int): 出力層のユニット数(次元)
        '''
        ###############層の数は適宜変更
        super().__init__()
        # 隠れ層：活性化関数はReLU
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        # 隠れ層：活性化関数はReLU
        self.fc1_2 = tf.keras.layers.Dense(hidden_dim2, activation='relu')
        # 隠れ層：活性化関数はReLU
        self.fc1_3 = tf.keras.layers.Dense(hidden_dim3, activation='relu')
        # 隠れ層：活性化関数はReLU
        self.fc1_4 = tf.keras.layers.Dense(hidden_dim4, activation='relu')
        """# 隠れ層：活性化関数はReLU
        self.fc1_5 = tf.keras.layers.Dense(hidden_dim5, activation='relu')
        # 隠れ層：活性化関数はReLU
        self.fc1_6 = tf.keras.layers.Dense(hidden_dim6, activation='relu')"""
        # 隠れ層：活性化関数はReLU
        #self.fc1_7 = tf.keras.layers.Dense(hidden_dim7, activation='relu')
        # 出力層：活性化関数はソフトマックス
        self.fc2 = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    @tf.function
    def call(self, x, training=None):
        '''MLPのインスタンスからコールバックされる関数
        
        Parameters: x(ndarray(float32)):訓練データ、または検証データ
        Returns(float32): MLPの出力として要素数3の1階テンソル        
        '''
        ###############層の数を変えた分だけ追加
        x = self.fc1(x) # 第1層の出力
        x = self.fc1_2(x) #第2層の出力
        x = self.fc1_3(x) #第3層の出力
        x = self.fc1_4(x) #第4層の出力
        """x = self.fc1_5(x) #第5層の出力
        x = self.fc1_6(x) #第6層の出力"""
        #x = self.fc1_7(x) #第7層の出力
        x = self.fc2(x) # 出力層の出力
        return x
#マルチラベル分類ではbinary_crossentropyを使うべし#############
loss_fn = tf.keras.losses.BinaryCrossentropy()

# 勾配降下アルゴリズムを使用するオプティマイザーを生成
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

import random as rd

rng = rd.SystemRandom()

# SINR計算関数
def calculate_sinr(d, desired_signal_power, interference_powers, noise_power, angles_rad, wavelength):
    # angles_rad: [θ_0, θ_1, ..., θ_N]
    # 干渉波の数
    N = len(interference_powers)
    # f_n0 = sin(θ_i) - sin(θ_0) for i=1..N
    f_n0 = np.sin(angles_rad[1:]) - np.sin(angles_rad[0])
    # 分子
    numerator = 2 * desired_signal_power * (
        np.sum(interference_powers * (1 - np.cos(2 * np.pi * f_n0 * d / wavelength)))
        + noise_power
    )
    # 干渉項
    interference_term = 2 * noise_power * np.sum(interference_powers)
    # クロースターム
    cross_term = 0.0
    for k in range(N-1):
        for l in range(k+1, N):
            cross_term += interference_powers[k] * interference_powers[l] * (
                1 - np.cos(2 * np.pi * (np.sin(angles_rad[k+1]) - np.sin(angles_rad[l+1])) * d / wavelength)
            )
    # 分母
    denominator = interference_term + 2 * cross_term + noise_power**2
    sinr = numerator / denominator
    # dB変換
    return 10 * np.log10(sinr)


# --- 定数・パラメータ ---
c = 3e8                        # 光速 [m/s]
f = 3e9                        # 周波数 [Hz]
wavelength = c / f             # 波長 [m]
num_antennas = 4               # アンテナ素子数
num_signals = 3                # 信号数 (所望波 + 干渉波数)
num_interf = num_signals - 1   # 干渉波数
snr_db = 20
noise_power = 10 ** (-snr_db / 10)
num_samples = 10000               # スナップショット数
num_pilot = 15
powers = np.array([1.0, 1.0, 1.0])  # 各信号電力
desired_signal_power = powers[0]
interference_powers = powers[1:]
antenna_pos = np.arange(num_antennas) * (wavelength / 2)

dd = np.arange(wavelength / 2, 100 * wavelength, 0.1 * wavelength)

#####for i in range(0,thetasize):
"------------------------到来角度---------------------"
filename = 4
new_model = tf.keras.models.load_model('/home/user_05/program/deeplearning_model/model_save_50per_3000node_inputRandangle_pilot_est_Python')

new_model.compile(metrics=['accuracy'])#optimizer='Adam',
                    #loss = 'BinaryCrossentropy',
                    #)
SINRpred = []
SINRoptimalpred = []

# 各素子間隔のカウント用配列を初期化
top50_counts = np.zeros(len(dd))
optimal_counts = np.zeros(len(dd))

desired_angle = []
times = []
for theta in tqdm(range(10000)):#00)):#0000)):
    # --- 到来角度生成 ---
    doa_deg = -90 + 180 * np.random.rand(num_signals)
    doa_rad = np.deg2rad(doa_deg)

    # --- ステアリングベクトル生成 ---
    sv = np.exp(-1j * 2 * np.pi * np.outer(antenna_pos, np.sin(doa_rad)) / wavelength)

    # --- 各信号生成（パイロット信号適用） ---
    #sig = (np.random.randn(num_signals, num_samples) + 1j * np.random.randn(num_signals, num_samples)) / np.sqrt(2)

    phase = np.random.uniform(0, 2 * np.pi, (num_signals, num_samples))
    sig = np.exp(1j * phase) * np.sqrt(powers)[:, None]  # (num_signals, num_samples)
    s_pilot = np.random.randint(0, 2, num_pilot) * 2 - 1  # {-1, 1}
    sig[0, :num_pilot] = s_pilot  # 所望波の先頭パイロット部を書き換え
    sig *= np.sqrt(powers)[:, None]

    # --- 雑音生成 ---
    noise = (np.random.randn(num_antennas, num_samples) + 1j * np.random.randn(num_antennas, num_samples)) / np.sqrt(2) * np.sqrt(noise_power)

    # --- 信号+雑音行列 X ---
    X = sv @ sig + noise

    # --- 相関行列Rの計算 ---
    R = (X @ X.conj().T) / num_samples

    # --- SVD & MUSIC ---
    U, _, _ = svd(R)
    En = U[:, num_signals:]  # 雑音サブスペース

    angles_grid = np.linspace(-90, 90, 1000)
    P_music = np.zeros_like(angles_grid, dtype=float)
    for k, ang in enumerate(angles_grid):
        a_test = np.exp(-1j * 2 * np.pi * antenna_pos * np.sin(np.deg2rad(ang)) / wavelength)
        P_music[k] = 1.0 / np.linalg.norm(En.conj().T @ a_test) ** 2
    P_music_db = 10 * np.log10(P_music / np.max(P_music))

    # --- MUSICピーク検出 ---
    peaks, _ = find_peaks(P_music_db)
    if len(peaks) < num_signals:
        continue
    peak_vals = P_music_db[peaks]
    top_idxs = peaks[np.argsort(peak_vals)[-num_signals:]]
    doa_est_deg = angles_grid[top_idxs]

    # --- パイロット信号処理 ---
    beamforming_weight = np.exp(-1j * 2 * np.pi * antenna_pos[:, np.newaxis] * np.sin(np.deg2rad(doa_est_deg))[np.newaxis, :] / wavelength)
    x_pilot = X[:, :num_pilot]  # Xの先頭num_pilot列
    rp = x_pilot @ s_pilot.conj()  # (4, num_pilot) x (num_pilot,) -> (4,)
    c = np.abs(beamforming_weight.conj().T @ rp)  # (3, 4) x (4,) -> (3,)

    idx = np.argmax(c)
    desired_theta = doa_est_deg[idx]
    desired_theta_rad = np.radians(desired_theta)

    # --- R_input作成（flatten）---
    first_row = R[0, :]  # shape (4,)
    R_input = []
    for j, val in enumerate(first_row):
        if j == 0:
            R_input.append(val)
        else:
            R_input.extend([val.real, val.imag])
    R_input = np.array(R_input, dtype=float).reshape(1, -1)
    R_input_norm = R_input / np.linalg.norm(R_input, axis=1, keepdims=True)
    """
    print(doa_deg)
    print(doa_est_deg)
    print(desired_theta)
    """
    # --- x の作成 ---
    #input_data2_sincos = np.column_stack([np.cos(desired_theta_rad), np.sin(desired_theta_rad)])
    #x = np.concatenate([R_input_norm, input_data2_sincos], axis=1)  # (1,9)
    x = np.concatenate([R_input, np.array(desired_theta_rad).reshape(1, 1)], axis=1)  # (1,8)
    #x = np.concatenate([R_input, np.array(doa_rad[0]).reshape(1, 1)], axis=1)  # (1,8)

    #################################ディープラーニング出力パラメータ（正解データ）###################################
    ################################################最適素子間隔####################################################

    "------Full search---------"
    # 各素子間隔に対してSINRを計算 (正確な角度の場合)
    sinr_over_true = []
    for d in dd:
        sinr = calculate_sinr(d, desired_signal_power, interference_powers, noise_power, doa_rad, wavelength)
        sinr_over_true.append(sinr)

    # SINRが最大になる素子間隔を取得 (正確な角度)
    sinr_dB_list = 10 * np.log10(sinr_over_true)
    optimal_index_true = np.argmax(sinr_over_true)
    optimal_d_true = dd[optimal_index_true]
    sinrdB_optimal = sinr_over_true[optimal_index_true]
    
    SINRoptimalpred.append(sinrdB_optimal)
    "-------Using deeplearning--------"
    

    pred = new_model.predict(x)#model(x_train, training = False)#model(x, training = False)#
    pred = np.array(pred)
    #print("predict....",pred)
    #print("predict[0]....",pred[0])
    #test_pred_rows, test_pred_columns = pred[0].shape
    pred0_sort = np.sort(pred[0])[::-1]
    pred0_argsort = np.argsort(-pred[0])#降順にSINRインデックス番号を返す※大きい順に[56,89,26,....]
    #percent = 20
    #pred0_sort_joui = pred0_sort[0 : int(percent)]  #並べ替えしてSINR自体の値を返す
    #pred0_joui = pred0_argsort[0 : int(percent)]    #並べ替えしたSINRの要素数：すなわち、これらを正解ラベルにしてその番号を1にする
    pred_No1 = pred0_argsort[0]
    #########################################################
    dd_joui2 = round(dd[pred_No1],2)      #素子間隔の値を返している 
    #print("dd_joui",dd_joui2)
    ######################################################
    "accuracy phase"
    SINRdB_deeplearning = calculate_sinr(dd_joui2, desired_signal_power, interference_powers, noise_power, doa_rad, wavelength)
    #SINR_deeplearning = np.array(SINR_deeplearning)
    print("SINRdB_dl",SINRdB_deeplearning)
    SINRpred.append(SINRdB_deeplearning)
    top50_indices = pred0_argsort[:50]   # 上位50個のインデックス
    top50_d_values = [round(dd[i], 2) for i in top50_indices]  # 素子間隔リスト


SINRpred = np.array(SINRpred)
SINRpred = SINRpred[np.isfinite(SINRpred)]  # NaNや±infを除く
SINRbins = 5000
SINR_pdf,SINR_yoko = np.histogram(SINRpred,bins=SINRbins)
print("SINRbins",SINRbins)
print("SINRpdf",SINR_pdf)
print("SINR_mean",SINRpred.mean())
# into CDF
cdf = np.cumsum(SINR_pdf) / np.sum(SINR_pdf)
bincenter = []
for a in range(SINRbins):
    SINR_bins_center = (SINR_yoko[a+1]+SINR_yoko[a])/2 #ヒストグラムの横軸棒の中央値
    bincenter.append(SINR_bins_center)
cdf=np.array(cdf)
bincenter = np.array(bincenter,dtype=float)
"相関行列R"#NN_spatial/R_10_test_random/
"""
with open('/home/user_05/program/cdf_d50_100lam_3000node/cdf_d50_100lam_3000node_Randangle_true_Python.csv', 'w', newline="") as f:
    writer = csv.writer(f)   #コンストラクタcsv.writer()の第一引数にopen()で開いたファイルオブジェクトを指定する。
    writer.writerow(bincenter)      #writerows()メソッドを使うと二次元配列（リストのリスト）を一気に書き込める
    writer.writerow(cdf)  
    #print("done")
"""

