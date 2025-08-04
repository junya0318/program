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

class_num = 995
data = 500000


lam = 0.1
dd = np.arange(lam/2,100*lam,0.01)

import random as rd

rng = rd.SystemRandom()

def SINRmmse(dAAAw):
    result1 = 2 * Parray[0] * (sum(Parray[n]*(1-np.cos(2*np.pi*dAAAw*\
        (np.sin(DoA[n]) - np.sin(DoA[0]))/lam)) for n in range(1,i+1)) + Pz)
    result2 = 2 * Pz * pn
    result3 = 2 * sum(sum(Parray[k]*Parray[l]*(1-(np.cos((np.sin(DoA[k])-np.sin(DoA[l]))\
        *2*np.pi*dAAAw/lam))) for l in range(k+1,i+1)) for k in range(1,i))
    
    result = result1/(result2+result3+(Pz**2))
    return result

def uwave(ndata):
    ran = np.random.uniform(0,1,ndata)
    phase = 2.0 * np.pi * ran #0~2πの一様乱数
    uwave = np.exp(1j * phase)
    return uwave

#-------normal random-------
def normal(Pn2,ndata):
	normal = np.random.normal(0.0, np.sqrt(Pn2/2) ,ndata) #/2正規分布 分散Pn
    #random.normal関数は任意の平均・標準偏差の正規分布 （「ガウス分布」とも言う） から乱数を生成する関数
	return normal

#-------normal random2-------
def normal2(Pn2,ndata):
	normal2 = np.random.normal(0.0, np.sqrt(Pn2/2) ,ndata) #/2正規分布 分散Pn
	return normal2

# ピーク検出（到来角のサイズに合わせる）
def detect_peaks(spectrum_dB, num_peaks):
    peaks, _ = find_peaks(spectrum_dB)
    peak_indices_sorted = sorted(peaks, key=lambda i: spectrum_dB[i], reverse=True)
    peak_indices_selected = peak_indices_sorted[:num_peaks]
    return angles[peak_indices_selected] if peak_indices_selected else np.array([])


"-----------------------素子移動回数--------------------"
"-----------------------素子移動回数--------------------"
element_step = 4 #何素子分とみなすか。#素子２を何回移動させるかは(element_step-2回)

"------------------信号生成数(ステップサイズ)------------"
ndata = 1000  #データ数(信号ステップサイズ)
ndata2 = 1000
"""
ndata3 = 1000  #データ数(信号ステップサイズ)
ndata4 = 1000

ndata5 = 1000  #データ数(信号ステップサイズ)
ndata6 = 1000

ndata7 = 1000  #データ数(信号ステップサイズ)
ndata8 = 1000
"""
tau_a = 500
tau_b = 2*tau_a
"""
tau_c = 3*tau_a
tau_d = 4*tau_a

tau_e = 5*tau_a
tau_f = 6*tau_a

tau_g = 7*tau_a
tau_h = 8*tau_a
"""
tau = [0, tau_a,tau_b]#, tau_c, tau_d]#, tau_e, tau_f, tau_g, tau_h]

"-------------------------波数設定----------------------"
i = 2 #干渉波数
print("干渉波数...",i)

"--------------------------素子数------------------------"
k = 2 #固定
print("素子数...",k)
K_L = element_step-i

"-------------------------電力----------------------"
Parray1 = [1 for s in range(i)]   #干渉波合計電力
pn = sum(Parray1[n] for n in range(0,i)) #変数i
p0 = [1]
Parray = np.hstack((p0, Parray1))
#print("Parray=",Parray[2])
Pz = 0.01

"------------------------その他パラメータ---------------------"
lam = 0.1
I = np.identity(element_step)
num_pilot = 15
num_antennas = 4
antenna_positions = np.arange(num_antennas) * (lam/2)
#####for i in range(0,thetasize):
"------------------------到来角度---------------------"
filename = 4
new_model = tf.keras.models.load_model('/home/user_05/program/model_save_50per_3000node_inputRandangle_pilot_ver2')

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
for theta in tqdm(range(10)):#00)):#0000)):
    start = time.time()
    """
    # ばらつきの大きさを設定（単位：度）
    variation_degrees = 5  # ±5度の範囲でばらつかせる

    # ラジアンへの変換
    variation_radians = np.radians(variation_degrees)

    # 基本角度（単位：度）
    thetas_degrees = 65
    theta1_degrees = 25
    theta2_degrees = -45
    
    # ラジアンに変換
    thetas = np.radians(thetas_degrees)# + rd.uniform(-variation_radians, variation_radians)
    theta1 = np.radians(theta1_degrees)# + rd.uniform(-variation_radians, variation_radians)
    theta2 = np.radians(theta2_degrees)# + rd.uniform(-variation_radians, variation_radians)
    """
    thetas =  rng.uniform(-np.pi/2,np.pi/2)
    theta1 =  rng.uniform(-np.pi/2,np.pi/2)
    theta2 =  rng.uniform(-np.pi/2,np.pi/2)
    
    # 結果を表示
    #print(f"Thetas (degrees): {np.degrees(thetas):.2f}")
    #print(f"Theta1 (degrees): {np.degrees(theta1):.2f}")
    #print(f"Theta2 (degrees): {np.degrees(theta2):.2f}")
    
            
    DoA = [thetas, theta1, theta2] #thetas所望波到来方向,theta1干渉波1到来方向,theta2干渉波1到来方向

    DoA_degrees = np.array([np.degrees(thetas), np.degrees(theta1), np.degrees(theta2)])
            
    #################################ディープラーニング入力パラメータ###################################
    ###########################################相関行列################################################

    #-------パラメータ初期設定------------
    d = [(lam/2)*(n+1) for n in range(0,element_step-1)]#入力パラメータ取得する際の素子間隔

    I = np.identity(element_step) #単位行列

    "----------------------------ステアリングベクトル----------------------------------------"
    Vl = [[np.exp(-1j * 2 * np.pi * (d[k] / lam) * np.sin(DoA[s])) for s in range(len(DoA))] for k in range(len(d))]#k素子目の方向ベクトル
            
    "--------------------------波生成＆相関行列作成---------------------------------------"
    wave = [uwave(ndata) for n in range(i+1)] #信号を数の分生成#waveからτ個ずらして取得したい。。。
            
    "--------------------------信号ずらしプログラム-------------------------------------"
    wave_2 = [uwave(ndata2) for n in range(i+1)]
    
    s_pilot = np.random.randint(0, 2, num_pilot) * 2 - 1  # {-1, 1}
    wave[0][:num_pilot] = s_pilot
    wave_2[0][:num_pilot] = s_pilot
    """
    wave_3 = [uwave(ndata3) for n in range(i+1)]
    wave_4 = [uwave(ndata4) for n in range(i+1)]
    
    wave_5 = [uwave(ndata5) for n in range(i+1)]
    wave_6 = [uwave(ndata6) for n in range(i+1)]
    
    wave_7 = [uwave(ndata7) for n in range(i+1)]
    wave_8 = [uwave(ndata8) for n in range(i+1)]
    """
    wave_hstack = np.hstack([wave_2,wave]) #waveにwave_2を結合wave_8,wave_7,wave_6,wave_5,wave_4,wave_3,
        
    wave_tau_a = wave_hstack[:, -(ndata+tau[1]):-tau_a] ###tau_a分ずらした信号
    wave_tau_b = wave_hstack[:, -(ndata+tau[2]):-tau_b] ###tau_b分ずらした信号
    """
    wave_tau_c = wave_hstack[:, -(ndata+tau[3]):-tau_c] ###tau_c分ずらした信号
    wave_tau_d = wave_hstack[:, -(ndata+tau[4]):-tau_d] ###tau_d分ずらした信号
    
    wave_tau_e = wave_hstack[:, -(ndata+tau[5]):-tau_e] 
    wave_tau_f = wave_hstack[:, -(ndata+tau[6]):-tau_f]
    
    wave_tau_g = wave_hstack[:, -(ndata+tau[7]):-tau_g] 
    wave_tau_h = wave_hstack[:, -(ndata+tau[8]):-tau_h] 
    """
    #print("wave_tau_a=",len(wave_tau_a[0,:]))

    sig = [np.sqrt(Parray[n])*wave[n] for n in range(i+1)] #波ランダム
    #print(Parray[0])
    #print("sig=",sig)
    sig_sum = sum(sig) #到来した信号を足し合わせる
    #print("sigsum=",sig_sum)

    nrm =normal(Pz,ndata) + 1j * normal(Pz,ndata)
    nrm2 =normal2(Pz,ndata) + 1j * normal2(Pz,ndata)

    sig_taua = [np.sqrt(Parray[n])*wave_tau_a[n] for n in range(i+1)] ###tau_a分ずらした信号
    sig_taua_sum = sum(sig_taua) #到来した信号を足し合わせる

    sig_taub = [np.sqrt(Parray[n])*wave_tau_b[n] for n in range(i+1)] ###tau_b分ずらした信号
    sig_taub_sum = sum(sig_taub) #到来した信号を足し合わせる

    """        
    sig_tauc = [np.sqrt(Parray[n])*wave_tau_c[n] for n in range(i+1)] ###tau_c分ずらした信号
    sig_tauc_sum = sum(sig_tauc) #到来した信号を足し合わせる

    sig_taud = [np.sqrt(Parray[n])*wave_tau_d[n] for n in range(i+1)] ###tau_a分ずらした信号
    sig_taud_sum = sum(sig_taud) #到来した信号を足し合わせる
    """
    """
    sig_taue = [np.sqrt(Parray[n])*wave_tau_e[n] for n in range(i+1)] 
    sig_taue_sum = sum(sig_taue) 

    sig_tauf = [np.sqrt(Parray[n])*wave_tau_f[n] for n in range(i+1)] 
    sig_tauf_sum = sum(sig_tauf) 
             
    sig_taug = [np.sqrt(Parray[n])*wave_tau_g[n] for n in range(i+1)] 
    sig_taug_sum = sum(sig_taug) 
            
    sig_tauh = [np.sqrt(Parray[n])*wave_tau_h[n] for n in range(i+1)] 
    sig_tauh_sum = sum(sig_tauh) 
    """
            
    "雑音ずらし"
    nrma =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrma2 = normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmb =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmb2 = normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    """
    nrmc =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmc2 = normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmd =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmd2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)
    """
    """
    nrme =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrme2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)
    nrmf =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmf2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)

    nrmg =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmg2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)
    nrmh =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmh2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)
    """
    nrm_hstack = np.hstack([nrmb,nrma,nrm]) #nrmh,nrmg,nrmf,nrme,nrmd,nrmc,
    nrm2_hstack = np.hstack([nrmb2,nrma2,nrm2]) #nrmh2,nrmg2,nrmf2,nrme2,nrmd2,nrmc2,
    nrm_tau_a = nrm_hstack[-(ndata+tau[1]):-tau[1]] ###nrmをtau_a分ずらした信号
    nrm2_tau_a = nrm2_hstack[-(ndata+tau[1]):-tau[1]] ###nrm2をtau_a分ずらした信号

    nrm_tau_b = nrm_hstack[-(ndata+tau[2]):-tau[2]] ###nrmをtau_b分ずらした信号
    nrm2_tau_b = nrm2_hstack[-(ndata+tau[2]):-tau[2]] ###nrm2をtau_b分ずらした信号
    """
    nrm_tau_c = nrm_hstack[-(ndata+tau[3]):-tau[3]] ###nrmをtau_c分ずらした信号
    nrm2_tau_c = nrm2_hstack[-(ndata+tau[3]):-tau[3]] ###nrm2をtau_c分ずらした信号

    nrm_tau_d = nrm_hstack[-(ndata+tau[4]):-tau[4]] ###nrmをtau_d分ずらした信号
    nrm2_tau_d = nrm2_hstack[-(ndata+tau[4]):-tau[4]] ###nrm2をtau_d分ずらした信号
    """
    """
    nrm_tau_e = nrm_hstack[-(ndata+tau[5]):-tau[5]] 
    nrm2_tau_e = nrm2_hstack[-(ndata+tau[5]):-tau[5]] 

    nrm_tau_f = nrm_hstack[-(ndata+tau[6]):-tau[6]] 
    nrm2_tau_f = nrm2_hstack[-(ndata+tau[6]):-tau[6]] 
    
    nrm_tau_g = nrm_hstack[-(ndata+tau[7]):-tau[7]] 
    nrm2_tau_g = nrm2_hstack[-(ndata+tau[7]):-tau[7]] 

    nrm_tau_h = nrm_hstack[-(ndata+tau[8]):-tau[8]] 
    nrm2_tau_h = nrm2_hstack[-(ndata+tau[8]):-tau[8]] 
    """
    "----------x1(t)作成-------"
    xin1 = sig_sum + nrm              #素子１に到来する信号

    "-------x1a(t+τa)作成------"
    xin1_taua = sig_taua_sum + nrm_tau_a 

    "-------x1b(t+τb)作成------"
    xin1_taub = sig_taub_sum + nrm_tau_b
    """
    "-------x1b(t+τc)作成------"
    xin1_tauc = sig_tauc_sum + nrm_tau_c

    "-------x1b(t+τb)作成------"
    xin1_taud = sig_taud_sum + nrm_tau_d
    """
    """
    xin1_taue = sig_taue_sum + nrm_tau_e 
    xin1_tauf = sig_tauf_sum + nrm_tau_f
    
    xin1_taug = sig_taug_sum + nrm_tau_g 
    xin1_tauh = sig_tauh_sum + nrm_tau_h
    """
    
    "----------x2(t)下準備----------"
    xinkb = Vl #素子kの方向ベクトル k素子の行,i波の列
    xinkbreshape = np.reshape(xinkb,(element_step-1,i+1)) #Vlを素子数-1×i+1の行列に変換
    #print("xinkbreshape",xinkbreshape)
    xinkbT = xinkbreshape.T

    "----------x2(t)作成-------"
    sig2 = [sig[n]*xinkbreshape[0,n] for n in range(i+1)] #素子2に到来する波だけでよい（素子2a,2bは↓に作成 ）
    sigreshape = np.reshape(sig2,(i+1,ndata))#sig2のarrayという文字を消した
    sig2_sum = sum(sigreshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2 = sig2_sum + nrm2       #素子kごとの入力

    "----------x2(t+τa)作成-------"
    sig2_taua = [sig_taua[n]*xinkbreshape[0,n] for n in range(i+1)] #素子2に到来する波だけでよい（素子2a,2bは↓に作成 ）
    sig2taua_reshape = np.reshape(sig2_taua,(i+1,ndata))#sig2のarrayという文字を消した
    sig2_taua_sum = sum(sig2taua_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2_taua = sig2_taua_sum + nrm_tau_a        #素子kごとの入力

    "----------x2(t+τb)作成-------"
    sig2_taub = [sig_taub[n]*xinkbreshape[0,n] for n in range(i+1)] #素子2に到来する波だけでよい（素子2a,2bは↓に作成 ）
    sig2taub_reshape = np.reshape(sig2_taub,(i+1,ndata))#sig2のarrayという文字を消した
    sig2_taub_sum = sum(sig2taub_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2_taub = sig2_taub_sum + nrm_tau_b     #素子kごとの入力
    """
    "----------x2(t+τc)作成-------"
    sig2_tauc = [sig_tauc[n]*xinkbreshape[0,n] for n in range(i+1)] #素子2に到来する波だけでよい（素子2a,2bは↓に作成 ）
    sig2tauc_reshape = np.reshape(sig2_tauc,(i+1,ndata))#sig2のarrayという文字を消した
    sig2_tauc_sum = sum(sig2tauc_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2_tauc = sig2_tauc_sum + nrm_tau_c     #素子kごとの入力

    "----------x2(t+τd)作成-------"
    sig2_taud = [sig_taud[n]*xinkbreshape[0,n] for n in range(i+1)] #素子2に到来する波だけでよい（素子2a,2bは↓に作成 ）
    sig2taud_reshape = np.reshape(sig2_taud,(i+1,ndata))#sig2のarrayという文字を消した
    sig2_taud_sum = sum(sig2taud_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2_taud = sig2_taud_sum + nrm_tau_d     #素子kごとの入力
    """
    """
    "--------x2(t+τe)作成-------"
    sig2_taue = [sig_taue[n]*xinkbreshape[0,n] for n in range(i+1)] #素子2に到来する波だけでよい（素子2a,2bは↓に作成 ）
    sig2taue_reshape = np.reshape(sig2_taud,(i+1,ndata))#sig2のarrayという文字を消した
    sig2_taue_sum = sum(sig2taue_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2_taue = sig2_taue_sum + nrm_tau_e     #素子kごとの入力

    "----------x2(t+τf)作成-------"
    sig2_tauf = [sig_tauf[n]*xinkbreshape[0,n] for n in range(i+1)] #素子2に到来する波だけでよい（素子2a,2bは↓に作成 ）
    sig2tauf_reshape = np.reshape(sig2_tauf,(i+1,ndata))#sig2のarrayという文字を消した
    sig2_tauf_sum = sum(sig2tauf_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2_tauf = sig2_tauf_sum + nrm_tau_f     #素子kごとの入力
    """
    "---------x2a(t+τa)作成--------"
    sig2a_taua = [sig_taua[n]*xinkbreshape[1,n] for n in range(i+1)] 
    sig2ataua_reshape = np.reshape(sig2a_taua,(i+1,ndata))#sig2のarrayという文字を消した
    sig2a_taua_sum = sum(sig2ataua_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2a_taua = sig2a_taua_sum + nrm2_tau_a      #素子kごとの入力

    "---------x2a(t+τb)作成--------"
    sig2a_taub = [sig_taub[n]*xinkbreshape[1,n] for n in range(i+1)] 
    sig2ataub_reshape = np.reshape(sig2a_taub,(i+1,ndata))#sig2のarrayという文字を消した
    sig2a_taub_sum = sum(sig2ataub_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2a_taub = sig2a_taub_sum + nrm_tau_b      #素子kごとの入力
    """
    "---------x2a(t+τc)作成--------"
    sig2a_tauc = [sig_tauc[n]*xinkbreshape[1,n] for n in range(i+1)] 
    sig2atauc_reshape = np.reshape(sig2a_tauc,(i+1,ndata))#sig2のarrayという文字を消した
    sig2a_tauc_sum = sum(sig2atauc_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2a_tauc = sig2a_tauc_sum + nrm_tau_c      #素子kごとの入力

    "---------x2a(t+τd)作成--------"
    sig2a_taud = [sig_taud[n]*xinkbreshape[1,n] for n in range(i+1)] 
    sig2ataud_reshape = np.reshape(sig2a_taud,(i+1,ndata))#sig2のarrayという文字を消した
    sig2a_taud_sum = sum(sig2ataud_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2a_taud = sig2a_taud_sum + nrm_tau_d      #素子kごとの入力
    """
    """
    "---------x2a(t+τe)作成--------"
    sig2a_taue = [sig_taue[n]*xinkbreshape[1,n] for n in range(i+1)] 
    sig2ataue_reshape = np.reshape(sig2a_taue,(i+1,ndata))#sig2のarrayという文字を消した
    sig2a_taue_sum = sum(sig2ataue_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2a_taue = sig2a_taue_sum + nrm_tau_e      #素子kごとの入力

    "---------x2a(t+τf)作成--------"
    sig2a_tauf = [sig_tauf[n]*xinkbreshape[1,n] for n in range(i+1)] 
    sig2atauf_reshape = np.reshape(sig2a_tauf,(i+1,ndata))#sig2のarrayという文字を消した
    sig2a_tauf_sum = sum(sig2atauf_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2a_tauf = sig2a_tauf_sum + nrm_tau_f      #素子kごとの入力
    """
    "----------x2b(t+τb)作成-------"
    sig2b_taub = [sig_taub[n]*xinkbreshape[2,n] for n in range(i+1)] 
    sig2btaub_reshape = np.reshape(sig2b_taub,(i+1,ndata))#sig2のarrayという文字を消した
    sig2b_taub_sum = sum(sig2btaub_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2b_taub = sig2b_taub_sum + nrm2_tau_b      #素子kごとの入力
    """
    "----------x2b(t+τc)作成-------"
    sig2b_tauc = [sig_tauc[n]*xinkbreshape[2,n] for n in range(i+1)] 
    sig2btauc_reshape = np.reshape(sig2b_tauc,(i+1,ndata))#sig2のarrayという文字を消した
    sig2b_tauc_sum = sum(sig2btauc_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2b_tauc = sig2b_tauc_sum + nrm_tau_c      #素子kごとの入力

    "----------x2b(t+τd)作成-------"
    sig2b_taud = [sig_taud[n]*xinkbreshape[2,n] for n in range(i+1)] 
    sig2btaud_reshape = np.reshape(sig2b_taud,(i+1,ndata))#sig2のarrayという文字を消した
    sig2b_taud_sum = sum(sig2btaud_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2b_taud = sig2b_taud_sum + nrm_tau_d      #素子kごとの入力
    """
    """
    "----------x2b(t+τe)作成-------"
    sig2b_taue = [sig_taue[n]*xinkbreshape[2,n] for n in range(i+1)] 
    sig2btaue_reshape = np.reshape(sig2b_taue,(i+1,ndata))#sig2のarrayという文字を消した
    sig2b_taue_sum = sum(sig2btaue_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2b_taue = sig2b_taue_sum + nrm_tau_e      #素子kごとの入力
    
    "----------x2b(t+τf)作成-------"
    sig2b_tauf = [sig_tauf[n]*xinkbreshape[2,n] for n in range(i+1)] 
    sig2btauf_reshape = np.reshape(sig2b_tauf,(i+1,ndata))#sig2のarrayという文字を消した
    sig2b_tauf_sum = sum(sig2btauf_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2b_tauf = sig2b_tauf_sum + nrm_tau_f      #素子kごとの入力
    """
    """
    "----------x2c(t+τc)作成-------"
    sig2c_tauc = [sig_tauc[n]*xinkbreshape[3,n] for n in range(i+1)] 
    sig2ctauc_reshape = np.reshape(sig2c_tauc,(i+1,ndata))#sig2のarrayという文字を消した
    sig2c_tauc_sum = sum(sig2ctauc_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2c_tauc = sig2c_tauc_sum + nrm2_tau_c      #素子kごとの入力

    "----------x2c(t+τd)作成-------"
    sig2c_taud = [sig_taud[n]*xinkbreshape[3,n] for n in range(i+1)] 
    sig2ctaud_reshape = np.reshape(sig2c_taud,(i+1,ndata))#sig2のarrayという文字を消した
    sig2c_taud_sum = sum(sig2ctaud_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2c_taud = sig2c_taud_sum + nrm_tau_c      #素子kごとの入力
    """
    """
    "----------x2c(t+τe)作成-------"
    sig2c_taue = [sig_taue[n]*xinkbreshape[3,n] for n in range(i+1)] 
    sig2ctaue_reshape = np.reshape(sig2c_taue,(i+1,ndata))#sig2のarrayという文字を消した
    sig2c_taue_sum = sum(sig2ctaue_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2c_taue = sig2c_taue_sum + nrm_tau_e      #素子kごとの入力

    "----------x2c(t+τf)作成-------"
    sig2c_tauf = [sig_tauf[n]*xinkbreshape[3,n] for n in range(i+1)] 
    sig2ctauf_reshape = np.reshape(sig2c_tauf,(i+1,ndata))#sig2のarrayという文字を消した
    sig2c_tauf_sum = sum(sig2ctauf_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2c_tauf = sig2c_tauf_sum + nrm_tau_f      #素子kごとの入力
    """
    """
    "----------x2d(t+τd)作成-------"
    sig2d_taud = [sig_taud[n]*xinkbreshape[4,n] for n in range(i+1)] 
    sig2dtaud_reshape = np.reshape(sig2d_taud,(i+1,ndata))#sig2のarrayという文字を消した
    sig2d_taud_sum = sum(sig2dtaud_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2d_taud = sig2d_taud_sum + nrm2_tau_d      #素子kごとの入力
    """
    """
    "----------x2d(t+τe)作成-------"
    sig2d_taue = [sig_taue[n]*xinkbreshape[4,n] for n in range(i+1)] 
    sig2dtaue_reshape = np.reshape(sig2d_taue,(i+1,ndata))#sig2のarrayという文字を消した
    sig2d_taue_sum = sum(sig2dtaue_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2d_taue = sig2d_taue_sum + nrm2_tau_e      #素子kごとの入力
    
    "----------x2d(t+τf)作成-------"
    sig2d_tauf = [sig_tauf[n]*xinkbreshape[4,n] for n in range(i+1)] 
    sig2dtauf_reshape = np.reshape(sig2d_tauf,(i+1,ndata))#sig2のarrayという文字を消した
    sig2d_tauf_sum = sum(sig2dtauf_reshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2d_tauf = sig2d_tauf_sum + nrm2_tau_f      #素子kごとの入力
    
    "----------x2e(t+τe)作成-------"
    sig2e_taue = [sig_taue[n]*xinkbreshape[5,n] for n in range(i+1)] 
    sig2etaue_reshape = np.reshape(sig2e_taue,(i+1,ndata))
    sig2e_taue_sum = sum(sig2etaue_reshape[n,:] for n in range(i+1))
    xin2e_taue = sig2e_taue_sum + nrm2_tau_e     
    
    "----------x2e(t+τf)作成-------"
    sig2e_tauf = [sig_tauf[n]*xinkbreshape[5,n] for n in range(i+1)] 
    sig2etauf_reshape = np.reshape(sig2e_tauf,(i+1,ndata))
    sig2e_tauf_sum = sum(sig2etauf_reshape[n,:] for n in range(i+1))
    xin2e_tauf = sig2e_tauf_sum + nrm2_tau_f     
    
    "----------x2f(t+τf)作成-------"
    sig2f_tauf = [sig_tauf[n]*xinkbreshape[6,n] for n in range(i+1)] 
    sig2ftauf_reshape = np.reshape(sig2f_tauf,(i+1,ndata))
    sig2f_tauf_sum = sum(sig2ftauf_reshape[n,:] for n in range(i+1))
    xin2f_tauf = sig2f_tauf_sum + nrm2_tau_f
    
    "----------x2g(t+τg)作成-------"      
    sig2g = [sig_taug[n]*xinkbreshape[7,n] for n in range(i+1)] 
    siggreshape = np.reshape(sig2g,(i+1,ndata))
    sig2g_sum = sum(siggreshape[n,:] for n in range(i+1))
    xin2_taug = sig2g_sum + nrm2_tau_g 

    "----------x2h(t+τh)作成-------"
    sig2h = [sig_tauh[n]*xinkbreshape[8,n] for n in range(i+1)] 
    sighreshape = np.reshape(sig2h,(i+1,ndata))
    sig2h_sum = sum(sighreshape[n,:] for n in range(i+1))
    xin2_tauh = sig2h_sum + nrm2_tau_h
    """


    "---Rxx,x11,x12,x13,x14のプログラム作成"
    xin1_tau = [xin1, xin1_taua, xin1_taub]#, xin1_tauc, xin1_taud]#, xin1_taue, xin1_tauf, xin1_taug, xin1_tauh]# , xin1_taug, xin1_tauh, xin1_taue, xin1_tauf
    xin1_tau_reshape = np.reshape(xin1_tau,(len(tau),ndata))
    xin1_tau_T = xin1_tau_reshape.T
    xin1_tau_conj = np.conj(xin1_tau_T)  #Xの複素共役転置
    xin1_tau_conj11 = np.reshape(xin1_tau_conj[:,0],(ndata,1))
    xin11 = np.dot(xin1_tau[0],xin1_tau_conj11) #dotなので信号全て足し合わせている
    #xin11real = xin11[0].real

    xin2_tau = [xin2, xin2a_taua, xin2b_taub]#, xin2c_tauc, xin2d_taud]#, xin2_taue, xin2_tauf, xin2_taug, xin2_tauh]#( x2,x3(t+τ3　),x4(t+τ4) ), xin2_taug, xin2_tauh, xin2_taue, xin2_tauf
    xin2_tau_reshape = np.reshape(xin2_tau,(len(tau),ndata))
    xin2_tau_T = xin2_tau_reshape.T
    xin2_tau_conj = np.conj(xin2_tau_T)  #Xの複素共役転置
    xin2_tau_conj22 = np.reshape(xin2_tau_conj[:,0],(ndata,1))
    xin2_tau_conj33 = np.reshape(xin2_tau_conj[:,1],(ndata,1))
    xin2_tau_conj44 = np.reshape(xin2_tau_conj[:,2],(ndata,1))
    """
    xin2_tau_conj55 = np.reshape(xin2_tau_conj[:,3],(ndata,1))
    xin2_tau_conj66 = np.reshape(xin2_tau_conj[:,4],(ndata,1))
    
    xin2_tau_conj77 = np.reshape(xin2_tau_conj[:,5],(ndata,1))
    xin2_tau_conj88 = np.reshape(xin2_tau_conj[:,6],(ndata,1))
    """

    xin22 = np.dot(xin2_tau[0],xin2_tau_conj22)
    xin33 = np.dot(xin2_tau[1],xin2_tau_conj33)
    xin44 = np.dot(xin2_tau[2],xin2_tau_conj44)
    """
    xin55 = np.dot(xin2_tau[3],xin2_tau_conj55)
    xin66 = np.dot(xin2_tau[4],xin2_tau_conj66)
    
    xin77 = np.dot(xin2_tau[5],xin2_tau_conj77)
    xin88 = np.dot(xin2_tau[6],xin2_tau_conj88)
    """

    xin12_tau_dot = np.dot(xin1_tau,xin2_tau_conj)#x1*x2,x1taua*x3,x1taub*x4
    # 1行目 
    xin1_2 = xin12_tau_dot[0, 0].item()
    xin1_3 = xin12_tau_dot[1, 1].item()
    xin1_4 = xin12_tau_dot[2, 2].item()
    """
    xin1_5 = xin12_tau_dot[3, 3].item()
    xin1_6 = xin12_tau_dot[4, 4].item()
    
    xin1_7 = xin12_tau_dot[5, 5].item()
    xin1_8 = xin12_tau_dot[6, 6].item()
    """
    # 2行目
    xin2_1 = np.conj(xin12_tau_dot[0, 0]).item()
    xin2_3 = np.dot(xin2_taua, xin2_tau_conj[:,1])
    xin2_4 = np.dot(xin2_taub, xin2_tau_conj[:,2])
    """
    xin2_5 = np.dot(xin2_tauc, xin2_tau_conj[:,3])
    xin2_6 = np.dot(xin2_taud, xin2_tau_conj[:,4])
    
    xin2_7 = np.dot(xin2_taue, xin2_tau_conj[:,5])
    xin2_8 = np.dot(xin2_tauf, xin2_tau_conj[:,6])
    """
    # 3行目
    xin3_1 = np.conj(xin1_3).item()
    xin3_2 = np.conj(xin2_3).item()
    xin3_4 = np.dot(xin2a_taub, xin2_tau_conj[:,2])
    """
    xin3_5 = np.dot(xin2a_tauc, xin2_tau_conj[:,3])
    xin3_6 = np.dot(xin2a_taud, xin2_tau_conj[:,4])
    
    xin3_7 = np.dot(xin2a_taue, xin2_tau_conj[:,5])
    xin3_8 = np.dot(xin2a_tauf, xin2_tau_conj[:,6])
    """
    # 4行目
    xin4_1 = np.conj(xin1_4).item()
    xin4_2 = np.conj(xin2_4).item()
    xin4_3 = np.conj(xin3_4).item()
    """
    xin4_5 = np.dot(xin2b_tauc, xin2_tau_conj[:,3])
    xin4_6 = np.dot(xin2b_taud, xin2_tau_conj[:,4])
    
    xin4_7 = np.dot(xin2b_taue, xin2_tau_conj[:,5])
    xin4_8 = np.dot(xin2b_tauf, xin2_tau_conj[:,6])
    """
    """
    # 5行目
    xin5_1 = np.conj(xin1_5).item()
    xin5_2 = np.conj(xin2_5).item()
    xin5_3 = np.conj(xin3_5).item()
    xin5_4 = np.conj(xin4_5).item()
    xin5_6 = np.dot(xin2c_taud, xin2_tau_conj[:,4])
    
    xin5_7 = np.dot(xin2c_taue, xin2_tau_conj[:,5])
    xin5_8 = np.dot(xin2c_tauf, xin2_tau_conj[:,6])
    """
    """
    # 6行目
    xin6_1 = np.conj(xin1_6).item()
    xin6_2 = np.conj(xin2_6).item()
    xin6_3 = np.conj(xin3_6).item()
    xin6_4 = np.conj(xin4_6).item()
    xin6_5 = np.conj(xin5_6).item()
    
    xin6_7 = np.dot(xin2d_taue, xin2_tau_conj[:,5])
    xin6_8 = np.dot(xin2d_tauf, xin2_tau_conj[:,6])
    """
    """       
    # 7行目
    xin7_1 = np.conj(xin1_7).item()
    xin7_2 = np.conj(xin2_7).item()
    xin7_3 = np.conj(xin3_7).item()
    xin7_4 = np.conj(xin4_7).item()
    xin7_5 = np.conj(xin5_7).item()
    xin7_6 = np.conj(xin6_7).item()
    xin7_8 = np.dot(xin2e_tauf, xin2_tau_conj[:,6])
    # 8行目
    xin8_1 = np.conj(xin1_8).item()
    xin8_2 = np.conj(xin2_8).item()
    xin8_3 = np.conj(xin3_8).item()
    xin8_4 = np.conj(xin4_8).item()
    xin8_5 = np.conj(xin5_8).item()
    xin8_6 = np.conj(xin6_8).item()
    xin8_7 = np.conj(xin7_8).item()
    """

    R = np.array([
            [xin11.item(), xin1_2, xin1_3, xin1_4],# xin1_5, xin1_6],# xin1_7, xin1_8],
            [xin2_1, xin22.item(), xin2_3, xin2_4],# xin2_5, xin2_6],# xin2_7, xin2_8],
            [xin3_1, xin3_2, xin33.item(), xin3_4],# xin3_5, xin3_6],# xin3_7, xin3_8],
            [xin4_1, xin4_2, xin4_3, xin44.item()]#, xin4_5, xin4_6],# xin4_7, xin4_8],
            #[xin5_1, xin5_2, xin5_3, xin5_4, xin55.item(), xin5_6],# xin5_7, xin5_8],
            #[xin6_1, xin6_2, xin6_3, xin6_4, xin6_5, xin66.item()]#, xin6_7, xin6_8],
            #[xin7_1, xin7_2, xin7_3, xin7_4, xin7_5, xin7_6, xin77.item(), xin7_8],
            #[xin8_1, xin8_2, xin8_3, xin8_4, xin8_5, xin8_6, xin8_7, xin88.item()]
        ], dtype=np.complex128)

    #print("xin1_tauc = ", xin1_tauc)
    #print("xin1_taud = ", xin1_taud)
    #print("xin1_tau = ", xin1_tau)
    #print("xin2_tau_conj", xin2_tau_conj)
    #print("xin1_5 =", xin1_5)
    #print("xin1_6 =", xin1_6)
    #print("xin12_tau_dot =", xin12_tau_dot)
    #print("対角成分 =", np.diag(xin12_tau_dot))

    
    # xin12_tau_dot の対角成分を抽出
    diagonal_elements = np.diag(xin12_tau_dot)

    # 実部・虚部を交互に格納
    real_imag_interleaved = np.ravel([[np.real(val), np.imag(val)] for val in diagonal_elements])

    # Rinput を構築
    Rinput = np.hstack([
        [np.real(xin11[0])],  # xin11 の最初の成分を追加
        real_imag_interleaved  # 実部と虚部を交互に格納した対角成分
    ])

    #Rinput2 = np.array([np.real(xin11[0]),np.real(xin1_2),np.imag(xin1_2),np.real(xin1_3),np.imag(xin1_3),np.real(xin1_4),np.imag(xin1_4),
                        #np.real(xin1_5),np.imag(xin1_5),np.real(xin1_6),np.imag(xin1_6)])#,np.real(xin1_7),np.imag(xin1_7),
                        #np.real(xin1_8),np.imag(xin1_8),np.real(xin1_9),np.imag(xin1_9),np.real(xin1_10),np.imag(xin1_10)])
    R_input = Rinput / ndata #ディープラーニング入力パラメータ
    #R_input2 = Rinput2 / ndata
    #print("R",R)
    #print("R2", R_input2)
    #print("確かめ用R3=",R3)

    # MUSIC スペクトルの計算
    angles = np.linspace(-90, 90, 1000)  # 角度範囲 (-90度から90度まで)
    music_spectrum = []
    music_spectrum_dB = []

    antenna_positions = np.arange(element_step) * lam / 2

    # MUSIC法の準備
    U, S, Vh = svd(R)
    noise_subspace = U[:, i+1:]

    for angle in angles:
        steering_vector_test = np.exp(-1j * 2 * np.pi * antenna_positions * np.sin(np.radians(angle)) / lam)
        music_spectrum.append(1 / np.linalg.norm(noise_subspace.conj().T @ steering_vector_test) ** 2)

    music_spectrum = 10 * np.log10(np.array(music_spectrum).flatten() / np.max(music_spectrum))
    detected_music_angles = detect_peaks(music_spectrum, i+1)
    
    # ピーク検出（上位3）
    peaks, _ = find_peaks(music_spectrum)
    if len(peaks) < i+1:
        continue
    # パイロット信号処理
    beamforming_weight = np.exp(
        -1j * 2 * np.pi * antenna_positions[:, np.newaxis] * np.sin(np.deg2rad(detected_music_angles))[np.newaxis, :] / lam
    )
    x_pilot = xin1[:num_pilot]
    x2_pilot = xin2[:num_pilot]
    x2a_pilot = xin2a_taua[:num_pilot]
    x2b_pilot = xin2b_taub[:num_pilot]
    X_pilot = np.vstack([x_pilot, x2_pilot, x2a_pilot, x2b_pilot])
    print(x_pilot)
    print(s_pilot)
    print(np.shape(X_pilot))
    rp = X_pilot @ s_pilot.conj().T
    print("beamforming_weight shape:", beamforming_weight.shape)
    print("rp shape:", np.shape(rp))
    c = np.abs(beamforming_weight.conj().T @ rp)
    idx = np.argmax(c)
    desired_theta = detected_music_angles[idx]

    desired_theta = np.array(desired_theta)
    desired_theta_rad = np.radians(desired_theta)
    
    print(DoA_degrees)
    print(detected_music_angles)
    print(desired_theta)
    """
    # 推定角度と真の到来角のペアリング
    sorted_music_detected_angles = np.full(len(DoA), np.nan)  # 初期値を NaN に

    for k, doa_val in enumerate(DoA):  # ループ変数を `doa_val` に変更
        if detected_music_angles.size == 0 or np.all(np.isnan(detected_music_angles)):
            break  # All detected angles have been used
        
        # Find the closest detected angle
        min_idx = np.nanargmin(np.abs(detected_music_angles - DoA_degrees[k]))
        sorted_music_detected_angles[k] = detected_music_angles[min_idx]
        
        # Mark the used detected angle as NaN
        detected_music_angles[min_idx] = np.nan
    """
    
    #print("Detected Angles(MUSIC):", sorted_music_detected_angles)
    #print("True Angles:", DoA_degrees)
    """
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(angles, music_spectrum, 'g', label='MUSIC')
    for angle in DoA_degrees:
        plt.axvline(x=angle, color='k', linestyle='--', label='True Angle' if angle == DoA_degrees[0] else "")
    for angle in sorted_music_detected_angles:
        plt.axvline(x=angle, color='m', linestyle='-.', label='Estimated Angle' if angle == sorted_music_detected_angles[0] else "")
    plt.xlabel('Angle (°)')
    plt.ylabel('Normalized Power (dB)')
    plt.title('Beamforming Methods Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"figure_beampatttern_theta{theta}.png")
    plt.close()
    """
    """
    # 3 波の到来方向推定が成功しているか確認
    valid_detections = np.count_nonzero(~np.isnan(sorted_music_detected_angles))
    
    if valid_detections == 3:  # 3 波が正しく推定された場合のみ SINR を計算
        print("All 3 signals detected. Proceeding with SINR calculation.")

        # SINRの値を保存するリスト
        sinr_values_true = np.zeros(len(dd))
        sinr_values_error = np.zeros(len(dd))

        # SINRを計算(真の到来方向)
        for k1, d in enumerate(dd):
            sinr_values_true[k1] = calculate_sinr(d, p0, Parray1, Pz, DoA, lam)

        # SINRを計算(到来方向推定)
        for k2, d in enumerate(dd):
            sinr_values_error[k2] = calculate_sinr(d, p0, Parray1, Pz, sorted_music_detected_angles, lam)

        # 最適な素子間隔を取得(真の到来方向)
        optimal_index_true = np.argmax(sinr_values_true)
        optimal_sinr_true = sinr_values_true[optimal_index_true]
        optimal_d_true = dd[optimal_index_true]

        # 最適な素子間隔を取得(到来方向推定)
        optimal_index_error = np.argmax(sinr_values_error)
        optimal_sinr_error = sinr_values_error[optimal_index_error]
        optimal_d_error = dd[optimal_index_error]

        # 最適な SINR と誤差を計算
        sinr_values_optimal_error = calculate_sinr(optimal_d_error, p0, Parray1, Pz, DoA, lam)

        # ここで追加（以前のデータを消さない）
        SINRdB_optimal.append(optimal_sinr_true)
        SINRdB_error.append(sinr_values_optimal_error)

        # 結果の表示
        print("SINR(到来方向推定):", sinr_values_optimal_error)
        print("SINR(真の到来方向):", optimal_sinr_true)
    else:
        print(f"Only {valid_detections} signals detected. Skipping SINR calculation.")
    """

    #################################ディープラーニング出力パラメータ（正解データ）###################################
    ################################################最適素子間隔####################################################
    """
    if valid_detections == 3:  # 3 波が正しく推定された場合のみ SINR を計算

        x = np.array(np.radians(sorted_music_detected_angles))

        # 入力形状を (1, 3) に修正
        x = x.reshape(1, -1)
    """

    "-----------compare phase---------"
    R_input_norm = R_input / np.linalg.norm(R_input, axis=1, keepdims=True)
    # 例: R_inputがshape (7,), desired_thetaがスカラー
    x = np.concatenate([R_input_norm], axis=1)
    print(x)  # (1, 8)


        #x = np.array(np.reshape(R_input,(1,2*filename-1)))
        #print("x",x[0:1])

    "------Full search---------"
    # 各素子間隔に対してSINRを計算 (正確な角度の場合)
    sinr_over_true = []
    for d in dd:
        sinr = SINRmmse(d)
        sinr_over_true.append(sinr)

    # SINRが最大になる素子間隔を取得 (正確な角度)
    sinr_dB_list = 10 * np.log10(sinr_over_true)
    optimal_index_true = np.argmax(sinr_over_true)
    optimal_d_true = dd[optimal_index_true]
    sinr_optimal = sinr_over_true[optimal_index_true]
    sinrdB_optimal = 10*np.log10(sinr_optimal) 
    
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
    times.append(time.time() - start)
    #print("dd_joui",dd_joui2)
    ######################################################
    "accuracy phase"
    SINR_deeplearning = SINRmmse(dd_joui2)
    #SINR_deeplearning = np.array(SINR_deeplearning)
    SINRdB_deeplearning = 10*np.log10(SINR_deeplearning) 
    print("SINRdB_dl",SINRdB_deeplearning)
    SINRpred.append(SINRdB_deeplearning)
    top50_indices = pred0_argsort[:50]   # 上位50個のインデックス
    top50_d_values = [round(dd[i], 2) for i in top50_indices]  # 素子間隔リスト


    for idx in top50_indices:
        top50_counts[idx] += 1

    # 最適素子間隔（1位）のみをカウント
    optimal_counts[pred0_argsort[0]] += 1


    os.makedirs("figures", exist_ok=True)
    """
    # グラフの描画（前のコードの続きに追加）
    plt.figure(figsize=(10, 6))
    plt.plot(dd/lam, sinr_dB_list, label='Full Search', linewidth=2)

    # Deep Learningの予測（1位）を赤点線で
    plt.axvline(dd_joui2/lam, color='r', linestyle='--', label=f'DL prediction\nd = {dd_joui2} m\nSINR = {SINRdB_deeplearning:.2f} dB')

    # 全探索での最適を緑点線で
    plt.axvline(optimal_d_true/lam, color='g', linestyle='--', label=f'Optimal\nd = {optimal_d_true} m\nSINR = {sinrdB_optimal:.2f} dB')

    # 上位50個の素子間隔にグレーの縦線を追加（薄く表示）
    for d_val in top50_d_values:
        plt.axvline(d_val/lam, color='gray', linestyle=':', alpha=0.5)

    # DoA（角度）の表示（右上に）
    DoA_str = '\n'.join([f'{i+1}: {round(deg, 1)}°' for i, deg in enumerate(DoA_degrees)])
    
    # グラフ右上の少し外に表示（axes座標で指定）
    plt.gca().annotate(
        f"Estimated DoAs\n{DoA_str}",
        xy=(1.02, 0.98),  # (x, y) 1.0 を超えると枠外
        xycoords='axes fraction',
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top',
        horizontalalignment='left'
    )

    # 軸ラベルなど
    plt.xlabel("Element Spacing d (m)", fontsize=14)
    plt.ylabel("SINR (dB)", fontsize=14)
    plt.title("SINR vs Element Spacing (Top 50 DL Predictions)", fontsize=16)
    plt.grid(True)
    #plt.legend(fontsize=12)
    # 横軸の目盛りを20λ刻みに設定（20λ, 40λ, 60λ, 80λ, 100λ）
    plt.xticks(np.arange(0, 101, 20), [f'{int(x)}λ' for x in np.arange(0, 101, 20)])
    plt.tight_layout()
    plt.tight_layout()

    plt.savefig(f"figure_DoAest_theta{theta}.png")
    plt.close()
    """
SINRpred = np.array(SINRpred)
end_time = time.time()

print(f"平均:", np.mean(times))
print("標準偏差", np.std(times))

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
with open('/home/user_05/program/cdf_d50_100lam_3000node/cdf_theta_est_0.5lam_input'+str(filename)+'.csv', 'w', newline="") as f:
    writer = csv.writer(f)   #コンストラクタcsv.writer()の第一引数にopen()で開いたファイルオブジェクトを指定する。
    writer.writerow(bincenter)      #writerows()メソッドを使うと二次元配列（リストのリスト）を一気に書き込める
    writer.writerow(cdf)  
    #print("done")
"""

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(bincenter,cdf)
#ax.plot(SINRpred,cdf ,label = 'CDF')

#ax.set_ylabel('SINR       (dB)',size=17)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=17)
#ax.set_xticks([lam/2,20,40,60,80,100,dd[dmax_SINR2]])#,dd[dmin_a12],
#------------------------拡大------------------------------------
#SINR
#ax.set_xticks([-5,0,5,10,15,20,25])
ax.set_xlim(0,25)#(dd[dmax_SINR2]-3,dd[dmax_SINR2]+3)
#ax.set_yticks([-50,-40,-30,-20,-10,0,0.5,1,10,20])
#ax.set_ylim(0,1)

#----------------------------------------------------------------
ax.grid()
#plt.xticks(rotation=35)
# ラベルの位置を揃える。
#fig.align_labels()
#plt.tight_layout()
#plt.xticks(rotation=33)
plt.show()