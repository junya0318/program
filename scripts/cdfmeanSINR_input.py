"""
accuracy making myself

"""
import tensorflow as tf
from tqdm import tqdm
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
import time
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

"-----------------------素子移動回数--------------------"
"-----------------------素子移動回数--------------------"
element_step = 6 #何素子分とみなすか。#素子２を何回移動させるかは(element_step-2回)

"------------------信号生成数(ステップサイズ)------------"
ndata = 10000  #データ数(信号ステップサイズ)
ndata2 = 10000
ndata3 = 10000  #データ数(信号ステップサイズ)
ndata4 = 10000
"""
ndata5 = 10000  #データ数(信号ステップサイズ)
ndata6 = 10000

ndata7 = 10000  #データ数(信号ステップサイズ)
ndata8 = 10000
"""
tau_a = 5000
tau_b = 2*tau_a
tau_c = 3*tau_a
tau_d = 4*tau_a
"""
tau_e = 5*tau_a
tau_f = 6*tau_a

tau_g = 7*tau_a
tau_h = 8*tau_a
"""
tau = [0, tau_a,tau_b, tau_c, tau_d]#, tau_e, tau_f]#, tau_g, tau_h]

"-------------------------波数設定----------------------"
i = 2 #干渉波数
print("干渉波数...",i)

"--------------------------素子数------------------------"
k = 2 #固定
print("素子数...",k)
K_L = element_step-i

"-------------------------電力----------------------"
Parray1 = [1/i for s in range(i)]   #干渉波合計電力
pn = sum(Parray1[n] for n in range(0,i)) #変数i
p0 = [1]
Parray = np.hstack((p0, Parray1))
#print("Parray=",Parray[2])
Pz = 0.01

"------------------------その他パラメータ---------------------"
lam = 0.1
I = np.identity(element_step)
#####for i in range(0,thetasize):
"------------------------到来角度---------------------"
filename = 6
new_model = tf.keras.models.load_model('/home/user_05/program/model_save_50per_6ele_0.5lam_3000node_ver3')

new_model.compile(metrics=['accuracy'])#optimizer='Adam',
                    #loss = 'BinaryCrossentropy',
                    #)
SINRpred = []
times = []
times_Rxx = []
times_pred= []
for theta in tqdm(range(100000)):#00)):#0000)):
    start = time.time() # 計測開始
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
    thetas = np.radians(thetas_degrees) + rd.uniform(-variation_radians, variation_radians)
    theta1 = np.radians(theta1_degrees) + rd.uniform(-variation_radians, variation_radians)
    theta2 = np.radians(theta2_degrees) + rd.uniform(-variation_radians, variation_radians)
    
    # 結果を表示
    #print(f"Thetas (degrees): {np.degrees(thetas):.2f}")
    #print(f"Theta1 (degrees): {np.degrees(theta1):.2f}")
    #print(f"Theta2 (degrees): {np.degrees(theta2):.2f}")
    """
    thetas =  rng.uniform(-np.pi/2,np.pi/2)
    theta1 =  rng.uniform(-np.pi/2,np.pi/2)
    theta2 =  rng.uniform(-np.pi/2,np.pi/2)
    
            
    DoA = [thetas , theta1, theta2] #thetas所望波到来方向,theta1干渉波1到来方向,theta2干渉波1到来方向
            
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
    wave_3 = [uwave(ndata3) for n in range(i+1)]
    wave_4 = [uwave(ndata4) for n in range(i+1)]
    """
    wave_5 = [uwave(ndata5) for n in range(i+1)]
    wave_6 = [uwave(ndata6) for n in range(i+1)]
    
    wave_7 = [uwave(ndata7) for n in range(i+1)]
    wave_8 = [uwave(ndata8) for n in range(i+1)]
    """
    wave_hstack = np.hstack([wave_4,wave_3,wave_2,wave]) #waveにwave_2を結合wave_8,wave_7,wave_6,wave_5,
        
    wave_tau_a = wave_hstack[:, -(ndata+tau[1]):-tau_a] ###tau_a分ずらした信号
    wave_tau_b = wave_hstack[:, -(ndata+tau[2]):-tau_b] ###tau_b分ずらした信号
    wave_tau_c = wave_hstack[:, -(ndata+tau[3]):-tau_c] ###tau_c分ずらした信号
    wave_tau_d = wave_hstack[:, -(ndata+tau[4]):-tau_d] ###tau_d分ずらした信号
    """
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
            
    sig_tauc = [np.sqrt(Parray[n])*wave_tau_c[n] for n in range(i+1)] ###tau_c分ずらした信号
    sig_tauc_sum = sum(sig_tauc) #到来した信号を足し合わせる

    sig_taud = [np.sqrt(Parray[n])*wave_tau_d[n] for n in range(i+1)] ###tau_a分ずらした信号
    sig_taud_sum = sum(sig_taud) #到来した信号を足し合わせる
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
    nrmc =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmc2 = normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmd =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmd2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)
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
    nrm_hstack = np.hstack([nrmd,nrmc,nrmb,nrma,nrm]) #nrmh,nrmg,nrmf,nrme,
    nrm2_hstack = np.hstack([nrmd2,nrmc2,nrmb2,nrma2,nrm2]) #nrmh2,nrmg2,nrmf2,nrme2
    nrm_tau_a = nrm_hstack[-(ndata+tau[1]):-tau[1]] ###nrmをtau_a分ずらした信号
    nrm2_tau_a = nrm2_hstack[-(ndata+tau[1]):-tau[1]] ###nrm2をtau_a分ずらした信号

    nrm_tau_b = nrm_hstack[-(ndata+tau[2]):-tau[2]] ###nrmをtau_b分ずらした信号
    nrm2_tau_b = nrm2_hstack[-(ndata+tau[2]):-tau[2]] ###nrm2をtau_b分ずらした信号

    nrm_tau_c = nrm_hstack[-(ndata+tau[3]):-tau[3]] ###nrmをtau_c分ずらした信号
    nrm2_tau_c = nrm2_hstack[-(ndata+tau[3]):-tau[3]] ###nrm2をtau_c分ずらした信号

    nrm_tau_d = nrm_hstack[-(ndata+tau[4]):-tau[4]] ###nrmをtau_d分ずらした信号
    nrm2_tau_d = nrm2_hstack[-(ndata+tau[4]):-tau[4]] ###nrm2をtau_d分ずらした信号
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

    "-------x1b(t+τb)作成------"
    xin1_tauc = sig_tauc_sum + nrm_tau_c

    "-------x1b(t+τb)作成------"
    xin1_taud = sig_taud_sum + nrm_tau_d
    """
    xin1_taue = sig_taue_sum + nrm_tau_e 
    xin1_tauf = sig_tauf_sum + nrm_tau_f
    
    xin1_taug = sig_taug_sum + nrm_tau_g 
    xin1_tauh = sig_tauh_sum + nrm_tau_h
    """

    "----------x2(t)下準備----------"
    xinkb = Vl #素子kの方向ベクトル k素子の行,i波の列
    xinkb = np.array(xinkb)  # リストをCuPy配列に変換
    xinkbreshape = np.reshape(xinkb,(element_step-1,i+1)) #Vlを素子数-1×i+1の行列に変換
    #print("xinkbreshape",xinkbreshape)
    xinkbT = xinkbreshape.T

    "----------x2(t)作成-------"
    sig2 = [sig[n]*xinkbreshape[0,n] for n in range(i+1)] #素子2に到来する波だけでよい（素子2a,2bは↓に作成 ）
    sigreshape = np.reshape(sig2,(i+1,ndata))#sig2のarrayという文字を消した
    sig2_sum = sum(sigreshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2 = sig2_sum + nrm2       #素子kごとの入力

    "---------x2a(t+τa)作成--------"
    sig2a = [sig_taua[n]*xinkbreshape[1,n] for n in range(i+1)] 
    sigareshape = np.reshape(sig2a,(i+1,ndata))#sig2のarrayという文字を消した
    sig2a_sum = sum(sigareshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2_taua = sig2a_sum + nrm2_tau_a      #素子kごとの入力

    "----------x2b(t+τb)作成-------"
    sig2b = [sig_taub[n]*xinkbreshape[2,n] for n in range(i+1)] 
    sigbreshape = np.reshape(sig2b,(i+1,ndata))#sig2のarrayという文字を消した
    sig2b_sum = sum(sigbreshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2_taub = sig2b_sum + nrm2_tau_b      #素子kごとの入力

    "----------x2c(t+τc)作成-------"
    sig2c = [sig_tauc[n]*xinkbreshape[3,n] for n in range(i+1)] 
    sigcreshape = np.reshape(sig2c,(i+1,ndata))#sig2のarrayという文字を消した
    sig2c_sum = sum(sigcreshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2_tauc = sig2c_sum + nrm2_tau_c      #素子kごとの入力

    "----------x2b(t+τd)作成-------"
    sig2d = [sig_taud[n]*xinkbreshape[4,n] for n in range(i+1)] 
    sigdreshape = np.reshape(sig2d,(i+1,ndata))#sig2のarrayという文字を消した
    sig2d_sum = sum(sigdreshape[n,:] for n in range(i+1))#各素子に到来する波を合成
    xin2_taud = sig2d_sum + nrm2_tau_d      #素子kごとの入力
    """
    sig2e = [sig_taue[n]*xinkbreshape[5,n] for n in range(i+1)] 
    sigereshape = np.reshape(sig2e,(i+1,ndata))
    sig2e_sum = sum(sigereshape[n,:] for n in range(i+1))
    xin2_taue = sig2e_sum + nrm2_tau_e      

    sig2f = [sig_tauf[n]*xinkbreshape[6,n] for n in range(i+1)] 
    sigfreshape = np.reshape(sig2f,(i+1,ndata))
    sig2f_sum = sum(sigfreshape[n,:] for n in range(i+1))
    xin2_tauf = sig2f_sum + nrm2_tau_f      
    
    sig2g = [sig_taug[n]*xinkbreshape[7,n] for n in range(i+1)] 
    siggreshape = np.reshape(sig2g,(i+1,ndata))
    sig2g_sum = sum(siggreshape[n,:] for n in range(i+1))
    xin2_taug = sig2g_sum + nrm2_tau_g      

    sig2h = [sig_tauh[n]*xinkbreshape[8,n] for n in range(i+1)] 
    sighreshape = np.reshape(sig2h,(i+1,ndata))
    sig2h_sum = sum(sighreshape[n,:] for n in range(i+1))
    xin2_tauh = sig2h_sum + nrm2_tau_h
    """
    "---Rxx,x11,x12,x13,x14のプログラム作成"
    xin1_tau = [xin1, xin1_taua, xin1_taub, xin1_tauc, xin1_taud]#, xin1_taue, xin1_tauf]#, xin1_taug, xin1_tauh]# , xin1_taug, xin1_tauh, xin1_taue, xin1_tauf
    xin1_tau_reshape = np.reshape(xin1_tau,(len(tau),ndata))
    xin1_tau_T = xin1_tau_reshape.T
    xin1_tau_conj = np.conj(xin1_tau_T)  #Xの複素共役転置
    xin1_tau_conj11 = np.reshape(xin1_tau_conj[:,0],(ndata,1))
    xin11 = np.dot(xin1_tau[0],xin1_tau_conj11) #dotなので信号全て足し合わせている
    #xin11real = xin11[0].real

    xin2_tau = [xin2, xin2_taua, xin2_taub, xin2_tauc, xin2_taud]#, xin2_taue, xin2_tauf]#, xin2_taug, xin2_tauh]#( x2,x3(t+τ3　),x4(t+τ4) ), xin2_taug, xin2_tauh, xin2_taue, xin2_tauf
    xin2_tau_reshape = np.reshape(xin2_tau,(len(tau),ndata))
    xin2_tau_T = xin2_tau_reshape.T
    xin2_tau_conj = np.conj(xin2_tau_T)  #Xの複素共役転置


    xin12_tau_dot = np.dot(xin1_tau,xin2_tau_conj)#x1*x2,x1taua*x3,x1taub*x4
    xin1_2 = xin12_tau_dot[0,0]
    xin1_3 = xin12_tau_dot[1,1]
    xin1_4 = xin12_tau_dot[2,2]
    #xin1_5 = xin12_tau_dot[3,3]
    #xin1_6 = xin12_tau_dot[4,4]
    #xin1_7 = xin12_tau_dot[5,5]
    #xin1_8 = xin12_tau_dot[6,6]
    #xin1_9 = xin12_tau_dot[7,7]
    #xin1_10 = xin12_tau_dot[8,8]
    
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
    #print("R",R_input)
    #print("R2", R_input2)
    #print("確かめ用R3=",R3)
    times_Rxx.append(time.time() - start)

    "-----------compare phase---------"
    x = np.array(np.reshape(R_input,(1,2*filename-1)))
    #print("x",x[0:1])
    "-------Using deeplearning--------"


    
    start_pred = time.time()
    pred = new_model.predict(x)#model(x_train, training = False)#model(x, training = False)#
    end_pred = time.time()
    times_pred.append(end_pred - start_pred)
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
    #print("SINRdB_dl",SINRdB_deeplearning)
    SINRpred.append(SINRdB_deeplearning)
SINRpred = np.array(SINRpred)

print(f"平均推論時間", np.mean(times_pred))
print(f"最大", np.max(times_pred))
print(f"最小", np.min(times_pred))

print(f"相関行列作成時間(6ele)", np.mean(times_Rxx))
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
with open('/home/user_05/program/cdf_d50_100lam_3000node/cdf_0.5lam_input'+str(filename)+'.csv', 'w', newline="") as f:
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