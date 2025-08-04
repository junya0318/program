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
    # result1の計算
    result1 = 2 * Parray[0] * (sum(Parray[n] * (1 - np.cos(2 * np.pi * dAAAw *
                   (np.sin(DoA[n]) - np.sin(DoA[0])) / lam)) for n in range(1, len(Parray))) + Pz)
    
    # result2の計算
    result2 = 2 * Pz * pn
    
    # result3の計算
    result3 = 2 * sum(sum(Parray[k] * Parray[l] * (1 - np.cos((np.sin(DoA[k]) - np.sin(DoA[l])) *
                   2 * np.pi * dAAAw / lam)) for l in range(k + 1, len(Parray))) for k in range(1, len(Parray) - 1))
    
    # SINRの最終計算
    result = result1 / (result2 + result3 + (Pz ** 2))
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

#--------信号を遅延する関数-----------------
def delay_signal(wave, delay_steps, ndata, tau):
    wave_hstack = np.hstack(wave)  # 波を結合
    return wave_hstack[:, -(ndata + tau): -tau]

"-----------------------素子移動回数--------------------"
element_step = 10 #何素子分とみなすか。#素子２を何回移動させるかは(element_step-2回)

"------------------信号生成数(ステップサイズ)------------"
ndata = 10000  #データ数(信号ステップサイズ，スナップショット数)
ndata2 = 10000 #信号をずらすために信号データ数を付け足す
ndata3 = 10000  
ndata4 = 10000
ndata5 = 10000  
ndata6 = 10000
ndata7 = 10000
ndata8 = 10000

tau_a = 5000 #信号をτ分ずらす
tau_b = 2*tau_a #2回目に動かす時にかかる時間τ2
tau_c = 3*tau_a #3回目に動かす時にかかる時間τ3
tau_d = 4*tau_a #3回目に動かす時にかかる時間τ4
tau_e = 5*tau_a
tau_f = 6*tau_a
tau_g = 7*tau_a
tau_h = 8*tau_a
# tau_e = 2*tau_a #3回目に動かす時にかかる時間τ3
tau = [0, tau_a, tau_b, tau_c, tau_d, tau_e, tau_f, tau_g, tau_h]

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

thetasize_desire =  10
thetasize_jammer1 = 30
thetasize_jammer2 = 10 
######----------↑結局3000000個のデータがほしいだけなのでdesireとかjammerとか関係ない-----------######

filename = 50


dd = np.arange(lam/2,100*lam,0.01) #素子間隔 範囲(lam/2~100λ) ステップサイズ0.1λ=0.01
# dd = np.arange(lam/2,100*lam,0.001) #素子間隔 範囲(lam/2~100λ) ステップサイズ0.01λ=0.001

for ll in tqdm(range(0,10)):
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
            Vl = [[np.exp(-1j * 2 * np.pi * (d[k] / lam) * np.sin(DoA[s])) for s in range(len(DoA))] for k in range(len(d))]#k素子目の方向ベクトル
            
            "--------------------------波生成＆相関行列作成---------------------------------------"
            wave = [uwave(ndata) for n in range(i+1)] #信号を数の分生成#waveからτ個ずらして取得したい。。。
            
            "--------------------------信号ずらしプログラム-------------------------------------"
            wave_2 = [uwave(ndata2) for n in range(i+1)]
            wave_3 = [uwave(ndata3) for n in range(i+1)]
            wave_4 = [uwave(ndata4) for n in range(i+1)]
            wave_5 = [uwave(ndata5) for n in range(i+1)]
            wave_6 = [uwave(ndata6) for n in range(i+1)]
            wave_7 = [uwave(ndata7) for n in range(i+1)]
            wave_8 = [uwave(ndata8) for n in range(i+1)]
            wave_hstack = np.hstack([wave_8,wave_7,wave_6,wave_5,wave_4,wave_3,wave_2,wave]) #waveにwave_2を結合
        
            wave_tau_a = wave_hstack[:, -(ndata+tau[1]):-tau_a] ###tau_a分ずらした信号
            wave_tau_b = wave_hstack[:, -(ndata+tau[2]):-tau_b] ###tau_b分ずらした信号
            wave_tau_c = wave_hstack[:, -(ndata+tau[3]):-tau_c] ###tau_c分ずらした信号
            wave_tau_d = wave_hstack[:, -(ndata+tau[4]):-tau_d] ###tau_d分ずらした信号
            wave_tau_e = wave_hstack[:, -(ndata+tau[5]):-tau_e] 
            wave_tau_f = wave_hstack[:, -(ndata+tau[6]):-tau_f]
            wave_tau_g = wave_hstack[:, -(ndata+tau[7]):-tau_g] 
            wave_tau_h = wave_hstack[:, -(ndata+tau[8]):-tau_h]  
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

            sig_taue = [np.sqrt(Parray[n])*wave_tau_e[n] for n in range(i+1)] 
            sig_taue_sum = sum(sig_taue) 

            sig_tauf = [np.sqrt(Parray[n])*wave_tau_f[n] for n in range(i+1)] 
            sig_tauf_sum = sum(sig_tauf) 
            
            sig_taug = [np.sqrt(Parray[n])*wave_tau_g[n] for n in range(i+1)] 
            sig_taug_sum = sum(sig_taug) 
            
            sig_tauh = [np.sqrt(Parray[n])*wave_tau_h[n] for n in range(i+1)] 
            sig_tauh_sum = sum(sig_tauh) 
            
            
            "雑音ずらし"
            nrma =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrma2 = normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmb =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmb2 = normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmc =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmc2 = normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmd =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmd2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)
            nrme =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrme2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)
            nrmf =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmf2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)
            nrmg =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmg2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)
            nrmh =normal(Pz,ndata2) + 1j * normal(Pz,ndata2)
            nrmh2 = normal2(Pz,ndata2) + 1j * normal2(Pz,ndata2)

            nrm_hstack = np.hstack([nrmh,nrmg,nrmf,nrme,nrmd,nrmc,nrmb,nrma,nrm])
            nrm2_hstack = np.hstack([nrmh2,nrmg2,nrmf2,nrme2,nrmd2,nrmc2,nrmb2,nrma2,nrm2])

            delay_signal()
            nrm_tau_a = nrm_hstack[-(ndata+tau[1]):-tau[1]] ###nrmをtau_a分ずらした信号
            nrm2_tau_a = nrm2_hstack[-(ndata+tau[1]):-tau[1]] ###nrm2をtau_a分ずらした信号

            nrm_tau_b = nrm_hstack[-(ndata+tau[2]):-tau[2]] ###nrmをtau_b分ずらした信号
            nrm2_tau_b = nrm2_hstack[-(ndata+tau[2]):-tau[2]] ###nrm2をtau_b分ずらした信号

            nrm_tau_c = nrm_hstack[-(ndata+tau[3]):-tau[3]] ###nrmをtau_c分ずらした信号
            nrm2_tau_c = nrm2_hstack[-(ndata+tau[3]):-tau[3]] ###nrm2をtau_c分ずらした信号

            nrm_tau_d = nrm_hstack[-(ndata+tau[4]):-tau[4]] ###nrmをtau_d分ずらした信号
            nrm2_tau_d = nrm2_hstack[-(ndata+tau[4]):-tau[4]] ###nrm2をtau_d分ずらした信号

            nrm_tau_e = nrm_hstack[-(ndata+tau[5]):-tau[5]] 
            nrm2_tau_e = nrm2_hstack[-(ndata+tau[5]):-tau[5]] 

            nrm_tau_f = nrm_hstack[-(ndata+tau[6]):-tau[6]] 
            nrm2_tau_f = nrm2_hstack[-(ndata+tau[6]):-tau[6]] 

            nrm_tau_g = nrm_hstack[-(ndata+tau[7]):-tau[7]] 
            nrm2_tau_g = nrm2_hstack[-(ndata+tau[7]):-tau[7]] 

            nrm_tau_h = nrm_hstack[-(ndata+tau[8]):-tau[8]] 
            nrm2_tau_h = nrm2_hstack[-(ndata+tau[8]):-tau[8]] 

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

            xin1_taue = sig_taue_sum + nrm_tau_e 
            xin1_tauf = sig_tauf_sum + nrm_tau_f
            xin1_taug = sig_taug_sum + nrm_tau_g 
            xin1_tauh = sig_tauh_sum + nrm_tau_h


            "----------x2(t)下準備----------"
            xinkb = Vl #素子kの方向ベクトル k素子の行,i波の列
            xinkb = np.array(xinkb)  # リストをCuPy配列に変換
            xinkbreshape = np.reshape(xinkb,(element_step-1,i+1)) #Vlを素子数-1×i+1の行列に変換
            #print("xinkbreshape",xinkbreshape)
            xinkbT = xinkbreshape.T

            "----------x2(t)作成-------"
            sig2 = [sig[n]*xinkbreshape[0,n] for n in range(i+1)] #素子2に到来する波だけでよい（素子2a,2bは↓に作成 ）
            sig2 = np.array(sig2)  # リストをCuPy配列に変換
            sigreshape = np.reshape(sig2,(i+1,ndata))#sig2のarrayという文字を消した
            sig2_sum = sum(sigreshape[n,:] for n in range(i+1))#各素子に到来する波を合成
            xin2 = sig2_sum + nrm2       #素子kごとの入力

            "---------x2a(t+τa)作成--------"
            sig2a = [sig_taua[n]*xinkbreshape[1,n] for n in range(i+1)] 
            sig2a = np.array(sig2a)  # リストをCuPy配列に変換
            sigareshape = np.reshape(sig2a,(i+1,ndata))#sig2のarrayという文字を消した
            sig2a_sum = sum(sigareshape[n,:] for n in range(i+1))#各素子に到来する波を合成
            xin2_taua = sig2a_sum + nrm2_tau_a      #素子kごとの入力

            "----------x2b(t+τb)作成-------"
            sig2b = [sig_taub[n]*xinkbreshape[2,n] for n in range(i+1)] 
            sig2b = np.array(sig2b)  # リストをCuPy配列に変換
            sigbreshape = np.reshape(sig2b,(i+1,ndata))#sig2のarrayという文字を消した
            sig2b_sum = sum(sigbreshape[n,:] for n in range(i+1))#各素子に到来する波を合成
            xin2_taub = sig2b_sum + nrm2_tau_b      #素子kごとの入力
 
            "----------x2c(t+τc)作成-------"
            sig2c = [sig_tauc[n]*xinkbreshape[3,n] for n in range(i+1)] 
            sig2c = np.array(sig2c)  # リストをCuPy配列に変換
            sigcreshape = np.reshape(sig2c,(i+1,ndata))#sig2のarrayという文字を消した
            sig2c_sum = sum(sigcreshape[n,:] for n in range(i+1))#各素子に到来する波を合成
            xin2_tauc = sig2c_sum + nrm2_tau_c      #素子kごとの入力

            "----------x2b(t+τd)作成-------"
            sig2d = [sig_taud[n]*xinkbreshape[4,n] for n in range(i+1)] 
            sig2d = np.array(sig2d)  # リストをCuPy配列に変換
            sigdreshape = np.reshape(sig2d,(i+1,ndata))#sig2のarrayという文字を消した
            sig2d_sum = sum(sigdreshape[n,:] for n in range(i+1))#各素子に到来する波を合成
            xin2_taud = sig2d_sum + nrm2_tau_d      #素子kごとの入力

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
            
            "---Rxx,x11,x12,x13,x14のプログラム作成"
            xin1_tau = [xin1, xin1_taua, xin1_taub, xin1_tauc, xin1_taud, xin1_taue, xin1_tauf, xin1_taug, xin1_tauh]
            xin1_tau = np.array(xin1_tau)
            xin1_tau_reshape = np.reshape(xin1_tau,(len(tau),ndata))
            xin1_tau_T = xin1_tau_reshape.T
            xin1_tau_conj = np.conj(xin1_tau_T)  #Xの複素共役転置
            xin1_tau_conj11 = np.reshape(xin1_tau_conj[:,0],(ndata,1))
            xin11 = np.dot(xin1_tau[0],xin1_tau_conj11) #dotなので信号全て足し合わせている
            #xin11real = xin11[0].real

            xin2_tau = [xin2, xin2_taua, xin2_taub, xin1_tauc, xin1_taud, xin2_taue, xin2_tauf, xin2_taug, xin2_tauh] #( x2,x3(t+τ3　),x4(t+τ4) )
            xin2_tau = np.array(xin2_tau)
            xin2_tau_reshape = np.reshape(xin2_tau,(len(tau),ndata))
            xin2_tau_T = xin2_tau_reshape.T
            xin2_tau_conj = np.conj(xin2_tau_T)  #Xの複素共役転置
            xin2_tau_conj = np.array(xin2_tau_conj)

            xin12_tau_dot = np.dot(xin1_tau,xin2_tau_conj)#x1*x2,x1taua*x3,x1taub*x4
            xin1_2 = xin12_tau_dot[0,0]
            xin1_3 = xin12_tau_dot[1,1]
            xin1_4 = xin12_tau_dot[2,2]
            xin1_5 = xin12_tau_dot[3,3]
            xin1_6 = xin12_tau_dot[4,4]
            xin1_7 = xin12_tau_dot[5,5]
            xin1_8 = xin12_tau_dot[6,6]
            xin1_9 = xin12_tau_dot[7,7]
            xin1_10 = xin12_tau_dot[8,8]

            print("xin12_tau_dot =", xin12_tau_dot)
            print("対角成分 =", np.diag(xin12_tau_dot))
            
            Rinput = np.array([np.real(xin11[0]),np.real(xin1_2),np.imag(xin1_2),np.real(xin1_3),np.imag(xin1_3),np.real(xin1_4),np.imag(xin1_4),
                               np.real(xin1_5),np.imag(xin1_5),np.real(xin1_6),np.imag(xin1_6),np.real(xin1_7),np.imag(xin1_7),
                               np.real(xin1_8),np.imag(xin1_8),np.real(xin1_9),np.imag(xin1_9),np.real(xin1_10),np.imag(xin1_10)])
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
            SINR = np.array(SINR)
            SINRdB = 10*np.log10(SINR)
            dmax_SINR = np.argmax(SINRdB)

            SINR_sort = np.sort(SINRdB)[::-1]
            SINR_argsort = np.argsort(-SINRdB)#降順にSINRインデックス番号を返す※大きい順に[56,89,26,....]
            percent = filename#100 * lam / 0.01 * 0.05
            #print(percent)
            SINR_sort_joui = SINR_sort[0 : int(percent)]  #並べ替えしてSINR自体の値を返す
            SINR_joui = SINR_argsort[0 : int(percent)]    #並べ替えしたSINRの要素数：すなわち、これらを正解ラベルにしてその番号を1にする
            dd_joui = [round(float(dd[i]), 2) for i in SINR_joui]  # dd[i] を float に変換してから round() を適用
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
    output_dir = r'/home/user_05/program/R_d50per_random_10ele'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    file_path = os.path.join(output_dir, 'R_' + str(filename) + '_' + str(ll) + '.csv')
    with open(file_path, 'w', newline="") as f:
     writer = csv.writer(f)
     writer.writerows(Rlist)

    # 最適素子間隔の書き込み
    output_dir = r'/home/user_05/program/d_50per_10ele'
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