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
import pandas as pd
import os
import time
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


# 使用可能なGPUを取得
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # 特定のGPUを指定 (例: GPU 1を使用)
        tf.config.set_visible_devices(gpus[1], 'GPU')
        print(f"Using GPU: {gpus[1].name}")
        
        # メモリの成長を許可（推奨設定）
        tf.config.experimental.set_memory_growth(gpus[1], True)
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

# 標準化関数
def standardize(data, mean_values, std_values):
    return (data - mean_values) / (std_values + 1e-8)
"-----------------------素子移動回数--------------------"
"-----------------------素子移動回数--------------------"
element_step = 4 #何素子分とみなすか。#素子２を何回移動させるかは(element_step-2回)

"------------------信号生成数(ステップサイズ)------------"
ndata = 10000  #データ数(信号ステップサイズ)
ndata2 = 10000
tau_a = 5000
tau_b = 2*tau_a
tau = [0, tau_a,tau_b]#, tau_c, tau_d]#, tau_e, tau_f]# , tau_g, tau_h]

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
filename = 4
new_model = tf.keras.models.load_model('/home/user_05/program/deeplearning_model/model_save_50per_3000node_MATLAB_0.1lam_sna10000_2ele_compare')

new_model.compile(metrics=['accuracy'])#optimizer='Adam',
                    #loss = 'BinaryCrossentropy',
                    #)

"""
# ファイルのパス
file_path = '/home/user_05/program/R_d50per_random_90/R_50_0.csv'
data = pd.read_csv(file_path, header=None)

# データの特徴量を計算
data_features = {
    "Mean": data.mean(),
    "Standard Deviation": data.std(),
    "Min": data.min(),
    "Max": data.max(),
    "Median": data.median(),
    "Range": data.max() - data.min(),
    "Skewness": data.skew(),
    "Kurtosis": data.kurt()
}
"""
#mean_values = data.mean().values.reshape(1, -1)  # (1, 7)
#std_values = data.std().values.reshape(1, -1)    # (1, 7)

SINRpred = []
SINRoptimalpred=[]
times = []
times_Rxx = []
times_pred = []
# 各素子間隔のカウント用配列を初期化
top50_counts = np.zeros(len(dd))
optimal_counts = np.zeros(len(dd))

for theta in tqdm(range(100)):#00)):#0000)):
    
    start = time.time() # 計測開始
    """
    # ばらつきの大きさを設定（単位：度）
    variation_degrees = 5  # ±5度の範囲でばらつかせる

    # ラジアンへの変換
    variation_radians = np.radians(variation_degrees)

    # 基本角度（単位：度）
    thetas_degrees = 85
    theta1_degrees = 80
    theta2_degrees = -45

    # ラジアンに変換
    thetas = np.radians(thetas_degrees)# + rd.uniform(-variation_radians, variation_radians)
    theta1 = np.radians(theta1_degrees)# + rd.uniform(-variation_radians, variation_radians)
    theta2 = np.radians(theta2_degrees)# + rd.uniform(-variation_radians, variation_radians)
    """
    # 結果を表示
    #print(f"Thetas (degrees): {np.degrees(thetas):.2f}")
    #print(f"Theta1 (degrees): {np.degrees(theta1):.2f}")
    #print(f"Theta2 (degrees): {np.degrees(theta2):.2f}")
    
    #thetas = rng.uniform(-np.pi/2,np.pi/2)
    #theta1 = rng.uniform(-np.pi/2,np.pi/2)
    #theta2 = rng.uniform(-np.pi/2,np.pi/2)
    
    #print("所望波DoA", thetas*(180/np.pi), "干渉波DoA=", theta1,theta2)
    
    # --- 到来角度生成 ---
    doa_deg = -90 + 180 * np.random.rand(i+1)
    DoA = np.deg2rad(doa_deg)

    #DoA = [thetas , theta1, theta2] #0,1,...波到来方向
    #print("DoA", DoA)
    #################################ディープラーニング入力パラメータ###################################
    ###########################################相関行列################################################

    #-------パラメータ初期設定------------
    d = [(lam/2)*(n+1) for n in range(0,element_step-1)]#+tau[n]*d_ran
            #print("d",d)


    Vl =  [[np.exp(-1j *2* np.pi * (d[k]/lam)*np.sin(DoA[s])) for s in range(i+1)] for k in range(len(d))]
                            
    wave = [uwave(ndata) for n in range(i+1)] 

    wave_2 = [uwave(ndata2) for n in range(i+1)]
    wave_hstack = np.hstack([wave_2,wave]) #wave_4,wave_3
    #print("aaa",wave_hstack)
    wave_tau_a = wave_hstack[:, -(ndata+tau[1]):-tau_a] 
    #print("wave_tau_a",wave_tau_a.shape)
    wave_tau_b = wave_hstack[:, -(ndata+tau[2]):-tau_b] 
    #print("wave_tau_a=",len(wave_tau_a[0,:]))

    sig = [np.sqrt(Parray[n])*wave[n] for n in range(i+1)] 
    #print(Parray[0])
    #print("sig=",sig)
    sig_sum = sum(sig) 
    #print("sigsum=",sig_sum)

    nrm =normal(Pz,ndata) + 1j * normal(Pz,ndata)
    nrm2 =normal2(Pz,ndata) + 1j * normal2(Pz,ndata)

    sig_taua = [np.sqrt(Parray[n])*wave_tau_a[n] for n in range(i+1)] 
    sig_taua_sum = sum(sig_taua) 

    sig_taub = [np.sqrt(Parray[n])*wave_tau_b[n] for n in range(i+1)] 
    sig_taub_sum = sum(sig_taub) 
            
    nrma =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrma2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)
    nrmb =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrmb2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)
    nrm_hstack = np.hstack([nrmb,nrma,nrm])#nrmg,nrmf,nrme,nrmd,nrmd,nrmc,
    nrm2_hstack = np.hstack([nrmb2,nrma2,nrm2])#nrmg2nrmf2,nrme2,nrmd2,nrmd2,nrmc2,
        
            
    nrm_tau_a = nrm_hstack[-(ndata+tau[1]):-tau[1]] 
    nrm2_tau_a = nrm2_hstack[-(ndata+tau[1]):-tau[1]] 
            
    nrm_tau_b = nrm_hstack[-(ndata+tau[2]):-tau[2]] 
    nrm2_tau_b = nrm2_hstack[-(ndata+tau[2]):-tau[2]] 

    xin1 = sig_sum + nrm              
    #print("sig_taua_sum",sig_taua_sum.shape)
    #print("nrm_tau_a",nrm_tau_a.shape)
    xin1_taua = sig_taua_sum + nrm_tau_a 
    xin1_taub = sig_taub_sum + nrm_tau_b

    xinkb = Vl 
    xinkbreshape = np.reshape(xinkb,(element_step-1,i+1)) 
    #print("xinkbreshape",xinkbreshape)
    xinkbT = xinkbreshape.T

    sig2 = [sig[n]*xinkbreshape[0,n] for n in range(i+1)]  	
    sigreshape = np.reshape(sig2,(i+1,ndata))
    sig2_sum = sum(sigreshape[n,:] for n in range(i+1))
    xin2 = sig2_sum + nrm2       

    sig2a = [sig_taua[n]*xinkbreshape[1,n] for n in range(i+1)] 
    sigareshape = np.reshape(sig2a,(i+1,ndata))
    sig2a_sum = sum(sigareshape[n,:] for n in range(i+1))
    xin2_taua = sig2a_sum + nrm2_tau_a      

    sig2b = [sig_taub[n]*xinkbreshape[2,n] for n in range(i+1)] 
    sigbreshape = np.reshape(sig2b,(i+1,ndata))
    sig2b_sum = sum(sigbreshape[n,:] for n in range(i+1))
    xin2_taub = sig2b_sum + nrm2_tau_b      
            
    xin1_tau = [xin1, xin1_taua,xin1_taub]#,xin1_tauc, xin1_taud]#,xin1_taue, xin1_tauf]#,xin1_taug, xin1_tauh]
    xin1_tau_reshape = np.reshape(xin1_tau,(len(tau),ndata))
    xin1_tau_T = xin1_tau_reshape.T
    xin1_tau_conj = np.conj(xin1_tau_T)  
    xin1_tau_conj11 = np.reshape(xin1_tau_conj[:,0],(ndata,1))
    xin11 = np.dot(xin1_tau[0],xin1_tau_conj11) 
    #xin11real = xin11[0].real

    xin2_tau = [xin2, xin2_taua, xin2_taub]#,xin2_tauc, xin2_taud]#,xin2_taue, xin2_tauf]#,xin2_taug, xin2_tauh] #( x2,x3(t+�3),x4(t+�4) )
    xin2_tau_reshape = np.reshape(xin2_tau,(len(tau),ndata))
    xin2_tau_T = xin2_tau_reshape.T
    xin2_tau_conj = np.conj(xin2_tau_T)  

    xin12_tau_dot = np.dot(xin1_tau,xin2_tau_conj)#x1*x2,x1taua*x3,x1taub*x4
    xin1_2 = xin12_tau_dot[0,0]
    xin1_3 = xin12_tau_dot[1,1]
    xin1_4 = xin12_tau_dot[2,2]
    Rinput = np.array([np.real(xin11[0]),np.real(xin1_2),np.imag(xin1_2),np.real(xin1_3),np.imag(xin1_3),np.real(xin1_4),np.imag(xin1_4)])
    R_input = Rinput / ndata #ディープラーニング入力パラメータ
    #print("R_input",R_input)
    times_Rxx.append(time.time() - start)

    "-----------compare phase---------"
    #R_input_norm = R_input / np.linalg.norm(R_input)
    #x = np.array(np.reshape(R_input_norm,(1,7)))
    x = np.array(np.reshape(R_input,(1,7)))
    #print("x",x[0:1])
    # 標準化
    #standardize(x, mean_values, std_values)
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
    
    
    #print("sinr_optimal:",sinrdB_optimal)

    
    
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

    # 軸ラベルなど
    plt.xlabel("Element Spacing d (m)", fontsize=14)
    plt.ylabel("SINR (dB)", fontsize=14)
    plt.title("SINR vs Element Spacing (Top 50 DL Predictions)", fontsize=16)
    plt.grid(True)
    #plt.legend(fontsize=12)
    # 横軸の目盛りを20λ刻みに設定（20λ, 40λ, 60λ, 80λ, 100λ）
    plt.xticks(np.arange(0, 101, 20), [f'{int(x)}λ' for x in np.arange(0, 101, 20)])
    plt.tight_layout()

    plt.savefig(f"figure_theta{theta}.png")
    plt.close()
    """
SINRpred = np.array(SINRpred)
SINRoptimalpred = np.array(SINRoptimalpred)

print(f"平均推論時間", np.mean(times_pred))
print(f"最大", np.max(times_pred))
print(f"最小", np.min(times_pred))

print(f"相関行列作成時間", np.mean(times_Rxx))
print(f"平均:", np.mean(times))
print("標準偏差", np.std(times))


# 正規化
top50_pdf = top50_counts / (np.sum(optimal_counts) * 50)  # 上位50の合計試行回数で正規化
optimal_pdf = optimal_counts / np.sum(optimal_counts)     # 最適の試行回数で正規化


SINRbins = 5000
SINR_pdf,SINR_yoko = np.histogram(SINRpred,bins=SINRbins)
print("SINRbins",SINRbins)
print("SINRpdf",SINR_pdf)
print("SINR_mean",SINRpred.mean())
print("SINR_optimal_mean",SINRoptimalpred.mean())
# into CDF
cdf = np.cumsum(SINR_pdf) / np.sum(SINR_pdf)
bincenter = []
for a in range(SINRbins):
    SINR_bins_center = (SINR_yoko[a+1]+SINR_yoko[a])/2 #ヒストグラムの横軸棒の中央値
    bincenter.append(SINR_bins_center)
cdf=np.array(cdf)
bincenter = np.array(bincenter,dtype=float)
"""
# プロット
plt.figure(figsize=(12, 6))
plt.bar(dd/lam, top50_pdf, width=(dd[1]-dd[0])/lam, alpha=0.6, label='Top 50 DL predictions PDF')
plt.xlabel("Element Spacing d (m)", fontsize=14)
plt.ylabel("Probability Density", fontsize=14)
plt.title("PDF of Element Spacing (DL Predictions)", fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.xticks(np.arange(0, 101, 20), [f'{int(x)}λ' for x in np.arange(0, 101, 20)])
plt.tight_layout()

plt.savefig(f"figure_DL_top50_pdf.png")
plt.close()

plt.figure(figsize=(12, 6))
plt.bar(dd/lam, optimal_pdf, width=(dd[1]-dd[0])/lam, alpha=0.6, color='red', label='Optimal DL prediction PDF')
plt.xlabel("Element Spacing d (m)", fontsize=14)
plt.ylabel("Probability Density", fontsize=14)
plt.title("PDF of Element Spacing (DL Predictions)", fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.xticks(np.arange(0, 101, 20), [f'{int(x)}λ' for x in np.arange(0, 101, 20)])
plt.tight_layout()

plt.savefig(f"figure_DL_optimal_pdf.png")
plt.close()
"""
"相関行列R"#NN_spatial/R_10_test_random/

with open('/home/user_05/program/cdf_d50_100lam_3000node/cdf_d50_100lam_3000node_0.1lam_2ele_compare.csv', 'w', newline="") as f:
    writer = csv.writer(f)   #コンストラクタcsv.writer()の第一引数にopen()で開いたファイルオブジェクトを指定する。
    writer.writerow(bincenter)      #writerows()メソッドを使うと二次元配列（リストのリスト）を一気に書き込める
    writer.writerow(cdf)  
    print("done")



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