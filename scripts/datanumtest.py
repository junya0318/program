"""-------------------------------------------------------------------------------------

SINR上位(filename)個取得するプログラム

ディープラーニング入力パラメータ　相関行列
ディープラーニング出力パラメータ　最適素子間隔の上位20個
のデータを到来角度を変えるごとにcsvに取得するプログラム

filename ・・・ 正解の個数
---------------------------------------------------------------------------------------"""

#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from tqdm import tqdm
import csv
import os
import sys

from regex import B

#%matplotlib inline

#%config InlineBackend.figure_formats = {'png', 'retina'}

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
#-------normal random2------
def normal2(Pn2,ndata):
	normal2 = np.random.normal(0.0, np.sqrt(Pn2/2) ,ndata) #/2正規分布 分散Pn
	return normal2

"-----------------------素子移動回数--------------------"
element_step = 4 #何素子分とみなすか。#素子２を何回移動させるかは(element_step-2回)

"------------------信号生成数(ステップサイズ)------------"
ndata = 10000  #データ数(信号ステップサイズ，スナップショット数)
ndata2 = 10000 #信号をずらすために信号データ数を付け足す
tau_a = 5000 #信号をτ分ずらす
tau_b = 2*tau_a #2回目に動かす時にかかる時間τ2
tau = [0, tau_a, tau_b]

"-------------------------波数設定----------------------"
i = 2 #干渉波数
"-------------------------電力----------------------"
Parray1 = [1/i for s in range(i)]  
pn = sum(Parray1[n] for n in range(0,i)) #干渉波合計電力
p0 = [1] #所望波入力電力
Parray = np.hstack((p0, Parray1))
Pz = 0.01

"------------------------その他パラメータ---------------------"
lam = 0.1
I = np.identity(element_step)
"------------------------到来角度---------------------"

thetasize_desire =  100
thetasize_jammer1 = 300 
thetasize_jammer2 = 100 
######----------↑結局3000000個のデータがほしいだけなのでdesireとかjammerとか関係ない-----------######

filename = 50


dd = np.arange(lam/2,100*lam,0.01) #素子間隔 範囲(lam/2~100λ) ステップサイズ0.1λ=0.01
#dd = np.arange(lam/2,100*lam,0.001) #素子間隔 範囲(lam/2~100λ) ステップサイズ0.01λ=0.001

for ll in tqdm(range(0,100)):
    Rlist = [] 
    dlist = []
    
    for mm in range(thetasize_jammer1):
        print('mm',mm)
        for nn in range(thetasize_jammer2):
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

            xin2_tau = [xin2, xin2_taua, xin2_taub] #( x2,x3(t+τ3　),x4(t+τ4) )
            xin2_tau_reshape = np.reshape(xin2_tau,(len(tau),ndata))
            xin2_tau_T = xin2_tau_reshape.T
            xin2_tau_conj = np.conj(xin2_tau_T)  #Xの複素共役転置

            xin12_tau_dot = np.dot(xin1_tau,xin2_tau_conj)#x1*x2,x1taua*x3,x1taub*x4
            xin1_2 = xin12_tau_dot[0,0]
            xin1_3 = xin12_tau_dot[1,1]
            xin1_4 = xin12_tau_dot[2,2]
            Rinput = np.array([np.real(xin11[0]),np.real(xin1_2),np.imag(xin1_2),np.real(xin1_3),np.imag(xin1_3),np.real(xin1_4),np.imag(xin1_4)])
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
            SINRdB = 10*np.log10(SINR)
            dmax_SINR = np.argmax(SINRdB)

            SINR_sort = np.sort(SINRdB)[::-1]
            SINR_argsort = np.argsort(-SINRdB)#降順にSINRインデックス番号を返す※大きい順に[56,89,26,....]
            percent = filename#100 * lam / 0.01 * 0.05
            #print(percent)
            SINR_sort_joui = SINR_sort[0 : int(percent)]  #並べ替えしてSINR自体の値を返す
            SINR_joui = SINR_argsort[0 : int(percent)]    #並べ替えしたSINRの要素数：すなわち、これらを正解ラベルにしてその番号を1にする
            dd_joui = [round(dd[i],2) for i in SINR_joui]          #素子間隔の値を返している 
            dlist.append(SINR_joui)
            
            SINRk = np.array(SINR)
            dlistk = list(dlist)
            # print("SINR_sort_joui",SINR_sort_joui)
            # print("dd_joui",dd_joui)
            
            dmin_SINR = np.argmin(SINRdB)

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
    output_dir = r'/home/user_05/program/R_d50per_random_90'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    file_path = os.path.join(output_dir, 'R_' + str(filename) + '_' + str(ll) + '.csv')
    with open(file_path, 'w', newline="") as f:
     writer = csv.writer(f)
     writer.writerows(Rlist)

    # 最適素子間隔の書き込み
    output_dir = r'/home/user_05/program/d_50per_90'
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
print("l",l)7y
"""