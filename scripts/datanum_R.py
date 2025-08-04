"""-------------------------------------------------------------------------------------

SINR上位(filename)個取得するプログラム

ディープラーニング入力パラメータ　相関行列
ディープラーニング出力パラメータ　最適素子間隔の上位20個
のデータを到来角度を変えるごとにcsvに取得するプログラム

filename ・・・ 正解の個数
---------------------------------------------------------------------------------------"""

#import tensorflow as tf
import cupy as cp
import matplotlib.pyplot as plt
import random as rd
from tqdm import tqdm
import csv
import os
import sys

from regex import B

#%matplotlib inline

#%config InlineBackend.figure_formats = {'png', 'retina'}

cp.cuda.Device(0).use()  # GPU デバイスを初期化

rng = rd.SystemRandom()

def SINRmmse(dAAAw):
    # result1の計算
    result1 = 2 * Parray[0] * (sum(Parray[n] * (1 - cp.cos(2 * cp.pi * dAAAw *
                   (cp.sin(DoA[n]) - cp.sin(DoA[0])) / lam)) for n in range(1, len(Parray))) + Pz)
    
    # result2の計算
    result2 = 2 * Pz * pn
    
    # result3の計算
    result3 = 2 * sum(sum(Parray[k] * Parray[l] * (1 - cp.cos((cp.sin(DoA[k]) - cp.sin(DoA[l])) *
                   2 * cp.pi * dAAAw / lam)) for l in range(k + 1, len(Parray))) for k in range(1, len(Parray) - 1))
    
    # SINRの最終計算
    result = result1 / (result2 + result3 + (Pz ** 2))
    return result


def uwave(ndata):
    ran = cp.random.uniform(0,1,ndata)
    phase = 2.0 * cp.pi * ran #0~2πの一様乱数
    uwave = cp.exp(1j * phase)
    return uwave

#-------normal random-------
def normal(Pn2,ndata):
	normal = cp.random.normal(0.0, cp.sqrt(Pn2/2) ,ndata) #/2正規分布 分散Pn
    #random.normal関数は任意の平均・標準偏差の正規分布 （「ガウス分布」とも言う） から乱数を生成する関数
	return normal
#-------normal random2------
def normal2(Pn2,ndata):
	normal2 = cp.random.normal(0.0, cp.sqrt(Pn2/2) ,ndata) #/2正規分布 分散Pn
	return normal2

"-----------------------素子移動回数--------------------"
element_step = 6 #何素子分とみなすか。#素子２を何回移動させるかは(element_step-2回)

"------------------信号生成数(ステップサイズ)------------"
ndata = 10000  #データ数(信号ステップサイズ，スナップショット数)
ndata2 = 10000 #信号をずらすために信号データ数を付け足す
ndata3 = 10000  
ndata4 = 10000

tau_a = 5000 #信号をτ分ずらす
tau_b = 2*tau_a #2回目に動かす時にかかる時間τ2
tau_c = 3*tau_a #3回目に動かす時にかかる時間τ3
tau_d = 4*tau_a #3回目に動かす時にかかる時間τ4
# tau_e = 2*tau_a #3回目に動かす時にかかる時間τ3
tau = [0, tau_a, tau_b, tau_c, tau_d]

"-------------------------波数設定----------------------"
i = 2 #干渉波数
"-------------------------電力----------------------"
Parray1 = [1/i for s in range(i)]  
pn = sum(Parray1[n] for n in range(0,i)) #干渉波合計電力
p0 = [1] #所望波入力電力
Parray = cp.hstack((p0, Parray1))
Pz = 0.01

"------------------------その他パラメータ---------------------"
lam = 0.1
I = cp.identity(element_step)
"------------------------到来角度---------------------"

thetasize_desire =  100
thetasize_jammer1 = 300 
thetasize_jammer2 = 100 
######----------↑結局3000000個のデータがほしいだけなのでdesireとかjammerとか関係ない-----------######

filename = 10


#dd = np.arange(lam/2,100*lam,0.01) #素子間隔 範囲(lam/2~100λ) ステップサイズ0.1λ=0.01
dd = cp.arange(lam/2,100*lam,0.01) #素子間隔 範囲(lam/2~100λ) ステップサイズ0.01λ=0.001

for ll in tqdm(range(0,100)):
    Rlist = [] 
    dlist = []
    
    for mm in range(thetasize_jammer1):
        print('mm',mm)
        for nn in range(thetasize_jammer2):
            thetas =  rng.uniform(-cp.pi/2,cp.pi/2)
            theta1 =  rng.uniform(-cp.pi/2,cp.pi/2)
            theta2 =  rng.uniform(-cp.pi/2,cp.pi/2)
            
            
            DoA = [thetas , theta1, theta2] #thetas所望波到来方向,theta1干渉波1到来方向,theta2干渉波1到来方向
            
            #################################ディープラーニング入力パラメータ###################################
            ###########################################相関行列################################################

            #-------パラメータ初期設定------------
            d = [(lam/2)*(n+1) for n in range(0,element_step-1)]#入力パラメータ取得する際の素子間隔

            I = cp.identity(element_step) #単位行列

            "----------------------------ステアリングベクトル----------------------------------------"
            Vl = [[cp.exp(-1j * 2 * cp.pi * (d[k] / lam) * cp.sin(DoA[s])) for s in range(len(DoA))] for k in range(len(d))]#k素子目の方向ベクトル
            
            "--------------------------波生成＆相関行列作成---------------------------------------"
            wave = [uwave(ndata) for n in range(i+1)] #信号を数の分生成#waveからτ個ずらして取得したい。。。
            
            "--------------------------信号ずらしプログラム-------------------------------------"
            wave_2 = [uwave(ndata2) for n in range(i+1)]
            wave_3 = [uwave(ndata3) for n in range(i+1)]
            wave_4 = [uwave(ndata4) for n in range(i+1)]
            wave_hstack = cp.hstack([wave_4,wave_3,wave_2,wave]) #waveにwave_2を結合

            wave_tau_a = wave_hstack[:, -(ndata+tau[1]):-tau_a] ###tau_a分ずらした信号
            wave_tau_b = wave_hstack[:, -(ndata+tau[2]):-tau_b] ###tau_b分ずらした信号
            wave_tau_c = wave_hstack[:, -(ndata+tau[3]):-tau_c] ###tau_c分ずらした信号
            wave_tau_d = wave_hstack[:, -(ndata+tau[4]):-tau_d] ###tau_d分ずらした信号
            #print("wave_tau_a=",len(wave_tau_a[0,:]))

            sig = [cp.sqrt(Parray[n])*wave[n] for n in range(len(Parray))] #波ランダム
            #print(Parray[0])
            #print("sig=",sig)
            sig_sum = sum(sig) #到来した信号を足し合わせる
            #print("sigsum=",sig_sum)

            nrm =normal(Pz,ndata) + 1j * normal(Pz,ndata)
            nrm2 =normal2(Pz,ndata) + 1j * normal2(Pz,ndata)

            sig_taua = [cp.sqrt(Parray[n])*wave_tau_a[n] for n in range(len(Parray))] ###tau_a分ずらした信号
            sig_taua_sum = sum(sig_taua) #到来した信号を足し合わせる

            sig_taub = [cp.sqrt(Parray[n])*wave_tau_b[n] for n in range(len(Parray))] ###tau_b分ずらした信号
            sig_taub_sum = sum(sig_taub) #到来した信号を足し合わせる
            
            sig_tauc = [cp.sqrt(Parray[n])*wave_tau_c[n] for n in range(len(Parray))] ###tau_c分ずらした信号
            sig_tauc_sum = sum(sig_tauc) #到来した信号を足し合わせる

            sig_taud = [cp.sqrt(Parray[n])*wave_tau_d[n] for n in range(len(Parray))] ###tau_a分ずらした信号
            sig_taud_sum = sum(sig_taud) #到来した信号を足し合わせる
            
            "雑音ずらし"
            nrma =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrma2 = normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmb =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmb2 = normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmc =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmc2 = normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmd =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmd2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)

            nrm_hstack = cp.hstack([nrmd,nrmc,nrmb,nrma,nrm])
            nrm2_hstack = cp.hstack([nrmd2,nrmc2,nrmb2,nrma2,nrm2])
            nrm_tau_a = nrm_hstack[-(ndata+tau[1]):-tau[1]] ###nrmをtau_a分ずらした信号
            nrm2_tau_a = nrm2_hstack[-(ndata+tau[1]):-tau[1]] ###nrm2をtau_a分ずらした信号

            nrm_tau_b = nrm_hstack[-(ndata+tau[2]):-tau[2]] ###nrmをtau_b分ずらした信号
            nrm2_tau_b = nrm2_hstack[-(ndata+tau[2]):-tau[2]] ###nrm2をtau_b分ずらした信号

            nrm_tau_c = nrm_hstack[-(ndata+tau[3]):-tau[3]] ###nrmをtau_c分ずらした信号
            nrm2_tau_c = nrm2_hstack[-(ndata+tau[3]):-tau[3]] ###nrm2をtau_c分ずらした信号

            nrm_tau_d = nrm_hstack[-(ndata+tau[4]):-tau[4]] ###nrmをtau_d分ずらした信号
            nrm2_tau_d = nrm2_hstack[-(ndata+tau[4]):-tau[4]] ###nrm2をtau_d分ずらした信号

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

            "----------x2(t)下準備----------"
            xinkb = cp.array(Vl) #素子kの方向ベクトル k素子の行,i波の列
            xinkbreshape = cp.reshape(xinkb,(element_step-1,len(Parray))) #Vlを素子数-1×i+1の行列に変換
            #print("xinkbreshape",xinkbreshape)
            xinkbT = xinkbreshape.T

            "----------x2(t)作成-------"
            sig2 = [sig[n]*xinkbreshape[0,n] for n in range(len(Parray))] #素子2に到来する波だけでよい（素子2a,2bは↓に作成 ）
            sig2 = cp.array(sig2)  # リストをCuPy配列に変換
            sigreshape = cp.reshape(sig2,(len(Parray),ndata))#sig2のarrayという文字を消した
            sig2_sum = sum(sigreshape[n,:] for n in range(len(Parray)))#各素子に到来する波を合成
            xin2 = sig2_sum + nrm2       #素子kごとの入力

            "---------x2a(t+τa)作成--------"
            sig2a = [sig_taua[n]*xinkbreshape[1,n] for n in range(len(Parray))] 
            sig2a = cp.array(sig2a)  # リストをCuPy配列に変換
            sigareshape = cp.reshape(sig2a,(len(Parray),ndata))#sig2のarrayという文字を消した
            sig2a_sum = sum(sigareshape[n,:] for n in range(len(Parray)))#各素子に到来する波を合成
            xin2_taua = sig2a_sum + nrm2_tau_a      #素子kごとの入力

            "----------x2b(t+τb)作成-------"
            sig2b = [sig_taub[n]*xinkbreshape[2,n] for n in range(len(Parray))] 
            sig2b = cp.array(sig2b)  # リストをCuPy配列に変換
            sigbreshape = cp.reshape(sig2b,(len(Parray),ndata))#sig2のarrayという文字を消した
            sig2b_sum = sum(sigbreshape[n,:] for n in range(len(Parray)))#各素子に到来する波を合成
            xin2_taub = sig2b_sum + nrm2_tau_b      #素子kごとの入力
 
            "----------x2c(t+τc)作成-------"
            sig2c = [sig_tauc[n]*xinkbreshape[2,n] for n in range(len(Parray))] 
            sig2c = cp.array(sig2c)  # リストをCuPy配列に変換
            sigcreshape = cp.reshape(sig2c,(len(Parray),ndata))#sig2のarrayという文字を消した
            sig2c_sum = sum(sigcreshape[n,:] for n in range(len(Parray)))#各素子に到来する波を合成
            xin2_tauc = sig2c_sum + nrm2_tau_c      #素子kごとの入力

            "----------x2b(t+τd)作成-------"
            sig2d = [sig_taud[n]*xinkbreshape[2,n] for n in range(len(Parray))] 
            sig2d = cp.array(sig2d)  # リストをCuPy配列に変換
            sigdreshape = cp.reshape(sig2d,(len(Parray),ndata))#sig2のarrayという文字を消した
            sig2d_sum = sum(sigdreshape[n,:] for n in range(len(Parray)))#各素子に到来する波を合成
            xin2_taud = sig2d_sum + nrm2_tau_d      #素子kごとの入力

            "---Rxx,x11,x12,x13,x14のプログラム作成"
            xin1_tau = [xin1, xin1_taua, xin1_taub, xin1_tauc, xin1_taud]
            xin1_tau = cp.array(xin1_tau)
            xin1_tau_reshape = cp.reshape(xin1_tau,(len(tau),ndata))
            xin1_tau_T = xin1_tau_reshape.T
            xin1_tau_conj = cp.conj(xin1_tau_T)  #Xの複素共役転置
            xin1_tau_conj11 = cp.reshape(xin1_tau_conj[:,0],(ndata,1))
            xin11 = cp.dot(xin1_tau[0],xin1_tau_conj11) #dotなので信号全て足し合わせている
            #xin11real = xin11[0].real

            xin2_tau = [xin2, xin2_taua, xin2_taub, xin1_tauc, xin1_taud] #( x2,x3(t+τ3　),x4(t+τ4) )
            xin2_tau = cp.array(xin2_tau)
            xin2_tau_reshape = cp.reshape(xin2_tau,(len(tau),ndata))
            xin2_tau_T = xin2_tau_reshape.T
            xin2_tau_conj = cp.conj(xin2_tau_T)  #Xの複素共役転置
            xin2_tau_conj = cp.array(xin2_tau_conj)

            print("Shape of xin1_tau:", xin1_tau.shape)
            print("Shape of xin2_tau_conj:", xin2_tau_conj.shape)


            xin12_tau_dot = cp.dot(xin1_tau,xin2_tau_conj)#x1*x2,x1taua*x3,x1taub*x4
            xin1_2 = xin12_tau_dot[0,0]
            xin1_3 = xin12_tau_dot[1,1]
            xin1_4 = xin12_tau_dot[2,2]
            xin1_5 = xin12_tau_dot[3,3]
            xin1_6 = xin12_tau_dot[4,4]
            
            Rinput = cp.array([cp.real(xin11[0]),cp.real(xin1_2),cp.imag(xin1_2),cp.real(xin1_3),cp.imag(xin1_3),cp.real(xin1_4),cp.imag(xin1_4),
                               cp.real(xin1_5),cp.imag(xin1_5),cp.real(xin1_6),cp.imag(xin1_6)])
            R_input = Rinput / ndata #ディープラーニング入力パラメータ
            #print("R",R_input)
            Rlist.append(R_input)
            #print("確かめ用R3=",R3)

            #################################ディープラーニング出力パラメータ（正解データ）###################################
            ################################################最適素子間隔####################################################
            """dd = []
            SINR = []
            for d1 in np.arange(lam/2,100*lam,0.01):
                SINR.append(SINRmmse(d1))
                dd.append(d1)"""

            SINR=[SINRmmse(i) for i in dd]
            SINR = cp.array(SINR)
            SINRdB = 10*cp.log10(SINR)
            dmax_SINR = cp.argmax(SINRdB)

            SINR_sort = cp.sort(SINRdB)[::-1]
            SINR_argsort = cp.argsort(-SINRdB)#降順にSINRインデックス番号を返す※大きい順に[56,89,26,....]
            percent = filename#100 * lam / 0.01 * 0.05
            #print(percent)
            SINR_sort_joui = SINR_sort[0 : int(percent)]  #並べ替えしてSINR自体の値を返す
            SINR_joui = SINR_argsort[0 : int(percent)]    #並べ替えしたSINRの要素数：すなわち、これらを正解ラベルにしてその番号を1にする
            dd_joui = [round(float(dd[i]), 2) for i in SINR_joui]  # dd[i] を float に変換してから round() を適用
            dlist.append(SINR_joui)
            
            SINRk = cp.array(SINR)
            dlistk = list(dlist)
            # print("SINR_sort_joui",SINR_sort_joui)
            # print("dd_joui",dd_joui)
            
            dmin_SINR = cp.argmin(SINRdB)

            "------------値表示-----------"
            # d2 = dd[dmax_SINR] #SINR最大の素子間隔
            # print('SINRdB=', SINRdB)
            # print('SINR_sort=', SINR_sort)
            # print('SINR_argsort=', SINR_argsort)
            # print('SINR_joui=', SINR_joui)
            # print("ddd",dd_joui)
            # print('ためしSINR', SINRdB[SINR_argsort[0]])
            # print('SINR=',SINRdB[dmax_SINR])
            # print('d=',d2)
            
    # print("Rlist",Rlist)
   
    "-----------------------csv書き込み------------------------------"
    # 相関行列の書き込み
    output_dir = r'/home/user_05/program/R_d10per_random_6ele'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    file_path = os.path.join(output_dir, 'R_' + str(filename) + '_' + str(ll) + '.csv')
    with open(file_path, 'w', newline="") as f:
     writer = csv.writer(f)
     writer.writerows(Rlist)

    # 最適素子間隔の書き込み
    output_dir = r'/home/user_05/program/d_10per_6ele'
    if not os.path.exists(output_dir):
     os.makedirs(output_dir)

    file_path = os.path.join(output_dir, 'd_' + str(filename) + '_' + str(ll) + '.csv')
    with open(file_path, 'w', newline="") as f:
     writer = csv.writer(f)
     writer.writerows(dlist)

    print(str(ll) + ' Done!')
"""
"-----------------------csv読み込み------------------------------"
"相関行列R"    
with open('/Users/ryusuke/OneDrive/spacial/Vscode/NN_spatial/R_d5per.csv') as f:
    reader = csv.reader(f)
    l = f.read()
print("l",l)

"最適素子間隔d"    
with open('/Users/ryusuke/OneDrive/spacial/Vscode/NN_spatial/d_5per.csv') as f:
    reader = csv.reader(f)
    l = f.read()
print("l",l)
"""