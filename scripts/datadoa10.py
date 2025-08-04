#import tensorflow as tf                                                                                                                                                                                                                                                                                                                                                                                                                                                   7777777777777777777777777777777777777777777777
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from tqdm import tqdm
import csv
import os

# 環境変数
rng = rd.SystemRandom()

# 必要な変数や定数の設定
element_step = 4
ndata = 10000
ndata2 = 10000
tau_a = 5000
tau_b = 2 * tau_a
tau = [0, tau_a, tau_b]

i = 2  # 干渉波数
Parray1 = [1 / i for s in range(i)]
pn = sum(Parray1[n] for n in range(0, i))
p0 = [1]
Parray = np.hstack((p0, Parray1))
Pz = 0.01
lam = 0.1
I = np.identity(element_step)
thetasize_jammer1 = 300
thetasize_jammer2 = 100
filename = 50

dd = np.arange(lam / 2, 100 * lam, 0.01)

# 関数の定義
def SINRmmse(dAAAw):
    result1 = 2 * Parray[0] * (sum(Parray[n]*(1 - np.cos(2 * np.pi * dAAAw * (np.sin(DoA[n]) - np.sin(DoA[0])) / lam)) for n in range(1, i+1)) + Pz)
    result2 = 2 * Pz * pn

    result3 = 2 * sum(sum(Parray[k] * Parray[l] * (1 - np.cos((np.sin(DoA[k]) - np.sin(DoA[l])) * 2 * np.pi * dAAAw / lam)) for l in range(k+1, i+1)) for k in range(1, i))
    return result1 / (result2 + result3 + (Pz ** 2))

def uwave(ndata):
    phase = 2.0 * np.pi * np.random.uniform(0, 1, ndata)
    return np.exp(1j * phase)

def normal(Pn2, ndata):
    return np.random.normal(0.0, np.sqrt(Pn2 / 2), ndata) + 1j * np.random.normal(0.0, np.sqrt(Pn2 / 2), ndata)


# データ生成部分
for ll in tqdm(range(0, 100)):
    Rlist = [] 
    dlist = []
    input_data = []  # 到来方向を保存するリスト
    
    for mm in range(thetasize_jammer1):
        for nn in range(thetasize_jammer2):
            thetas = rng.uniform(-np.pi / 2, np.pi / 2)
            theta1 = rng.uniform(-np.pi / 2, np.pi / 2)
            theta2 = rng.uniform(-np.pi / 2, np.pi / 2)
            DoA = [thetas, theta1, theta2]
            
            sin_angles = np.sin(DoA)
            cos_angles = np.cos(DoA)
            
            # sinとcosを1行に結合してフラットなデータを生成
            flattened_data = np.hstack((sin_angles, cos_angles))
            input_data.append(flattened_data)

            """dd = []
            SINR = []
            for d1 in np.arange(lam/2,100*lam,0.01):
                SINR.append(SINRmmse(d1))
                dd.append(d1)"""
            # 出力パラメータ（最適素子間隔）を生成
            SINR = [SINRmmse(d1) for d1 in dd]
            SINRdB = 10 * np.log10(SINR)
            SINR_sort = np.sort(SINRdB)[::-1]
            SINR_argsort = np.argsort(-SINRdB)
            SINR_joui = SINR_argsort[:filename]
            dlist.append(SINR_joui)
            
    # CSVファイルに保存

     # 到来方向の書き込み
    output_dir = r'/home/user_05/program/DoA_50per'
    if not os.path.exists(output_dir):
     os.makedirs(output_dir)

    file_path = os.path.join(output_dir, 'DoA_' + str(filename) + '_' + str(ll) + '.csv')
    with open(file_path, 'w', newline="") as f:
     writer = csv.writer(f)
     writer.writerows(input_data)

     # 最適素子間隔の書き込み
    output_dir = r'/home/user_05/program/d_50per'
    if not os.path.exists(output_dir):
     os.makedirs(output_dir)

    file_path = os.path.join(output_dir, 'd_' + str(filename) + '_' + str(ll) + '.csv')
    with open(file_path, 'w', newline="") as f:
     writer = csv.writer(f)
     writer.writerows(dlist)

    print(str(ll) + ' Done!')
