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

model = MLP(3000, 3000, 3000, 3000, 495)
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
element_step = 4 #何素子分とみなすか。#素子２を何回移動させるかは(element_step-2回)

"------------------信号生成数(ステップサイズ)------------"
ndata = 10000  #データ数(信号ステップサイズ)
ndata2 = 10000
tau_a = 5000
tau_b = 2*tau_a
tau = [0, tau_a, tau_b]

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
new_model = tf.keras.models.load_model('/home/user_05/program/model_save_datanum50per_3000node_75')#

new_model.compile(metrics=['accuracy'])#optimizer='Adam',
                    #loss = 'BinaryCrossentropy',
                    #)
SINRpred = []
for theta in tqdm(range(100000)):#00)):#0000)):
    thetas = rng.uniform(-np.pi/2,np.pi/2)
    theta1 = rng.uniform(-np.pi/2,np.pi/2)
    theta2 = rng.uniform(-np.pi/2,np.pi/2)
    #print("所望波DoA", thetas*(180/np.pi), "干渉波DoA=", theta1,theta2)

    DoA = [thetas , theta1, theta2] #0,1,...波到来方向
    #print("DoA", DoA)
    #################################ディープラーニング入力パラメータ###################################
    ###########################################相関行列################################################

    #-------パラメータ初期設定------------
    d = [(lam/2)*(n+1) for n in range(0,element_step-1)]#+tau[n]*d_ran
    #print("d",d)
    I = np.identity(element_step) #単位行列

    "----------------------------ステアリングベクトル----------------------------------------"
    Vl =  [[np.exp(-1j *2* np.pi * (d[k]/lam)*np.sin(DoA[s])) for s in range(i+1)] for k in range(len(d))]#k素子目の方向ベクトル

    "--------------------------波生成＆相関行列作成---------------------------------------"
    wave = [uwave(ndata) for n in range(i+1)] #信号を数の分生成#waveからτ個ずらして取得したい。。。

    "--------------------------信号ずらしプログラム-------------------------------------"
    wave_2 = [uwave(ndata2) for n in range(i+1)]
    wave_hstack = np.hstack([wave_2,wave]) #waveにwave_2を結合

    wave_tau_a = wave_hstack[:, -(ndata+tau[1]):-tau_a] ###tau_a分ずらした信号
    wave_tau_b = wave_hstack[:, -(ndata+tau[2]):-tau_b] ###tau_b分ずらした信号
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

    "雑音ずらし"
    nrma =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
    nrma2 = normal(Pz,ndata2) + 1j * normal(Pz,ndata2)

    nrm_hstack = np.hstack([nrma,nrm])
    nrm2_hstack = np.hstack([nrma2,nrm2])
    nrm_tau_a = nrm_hstack[-(ndata+tau[1]):-tau[1]] ###nrmをtau_a分ずらした信号
    nrm2_tau_a = nrm2_hstack[-(ndata+tau[1]):-tau[1]] ###nrm2をtau_a分ずらした信号

    nrm_tau_b = nrm_hstack[-(ndata+tau[2]):-tau[2]] ###nrmをtau_b分ずらした信号
    nrm2_tau_b = nrm2_hstack[-(ndata+tau[2]):-tau[2]] ###nrm2をtau_b分ずらした信号

    "----------x1(t)作成-------"
    xin1 = sig_sum + nrm              #素子１に到来する信号

    "-------x1a(t+τa)作成------"
    xin1_taua = sig_taua_sum + nrm_tau_a 
    "-------x1b(t+τb)作成------"
    xin1_taub = sig_taub_sum + nrm_tau_b

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

    "---Rxx,x11,x12,x13,x14のプログラム作成"
    xin1_tau = [xin1, xin1_taua, xin1_taub]
    xin1_tau_reshape = np.reshape(xin1_tau,(len(tau),ndata))
    xin1_tau_T = xin1_tau_reshape.T
    xin1_tau_conj = np.conj(xin1_tau_T)  #Xの複素共役転置
    xin1_tau_conj11 = np.reshape(xin1_tau_conj[:,0],(ndata,1))
    xin11 = np.dot(xin1_tau[0],xin1_tau_conj11) #dotなので信号全て足し合わせている
    #xin11real = xin11[0].real

    xin2_tau = [xin2, xin2_taua, xin2_taub] #( x2,x3(t+τ3),x4(t+τ4) )
    xin2_tau_reshape = np.reshape(xin2_tau,(len(tau),ndata))
    xin2_tau_T = xin2_tau_reshape.T
    xin2_tau_conj = np.conj(xin2_tau_T)  #Xの複素共役転置

    xin12_tau_dot = np.dot(xin1_tau,xin2_tau_conj)#x1*x2,x1taua*x3,x1taub*x4
    xin1_2 = xin12_tau_dot[0,0]
    xin1_3 = xin12_tau_dot[1,1]
    xin1_4 = xin12_tau_dot[2,2]
    Rinput = np.array([np.real(xin11[0]),np.real(xin1_2),np.imag(xin1_2),np.real(xin1_3),np.imag(xin1_3),np.real(xin1_4),np.imag(xin1_4)])
    R_input = Rinput / ndata #ディープラーニング入力パラメータ
    #print("R_input",R_input)


    "-----------compare phase---------"
    x = np.array(np.reshape(R_input,(1,7)))
    #print("x",x[0:1])
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
    SINR_deeplearning = SINRmmse(dd_joui2)
    #SINR_deeplearning = np.array(SINR_deeplearning)
    SINRdB_deeplearning = 10*np.log10(SINR_deeplearning) 
    #print("SINRdB_dl",SINRdB_deeplearning)
    SINRpred.append(SINRdB_deeplearning)
SINRpred = np.array(SINRpred)

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
with open('/home/user_05/program/cdf_d50_100lam/cdf_d_50_100lam_75.csv', 'w', newline="") as f:
    writer = csv.writer(f)   #コンストラクタcsv.writer()の第一引数にopen()で開いたファイルオブジェクトを指定する。
    writer.writerow(bincenter)      #writerows()メソッドを使うと二次元配列（リストのリスト）を一気に書き込める
    writer.writerow(cdf)  
    #print("done")



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