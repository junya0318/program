"""
weightが保存されて初期値として使われているのか確認する
素子間隔maxのもの1つだけを選ぶので、ソフトマックス関数でいいか
・入力層の正規化について調べる。
・損失関数について調べる
・これなんだろう↓　調べるべき
# カテゴリカルデータの精度を取得するオブジェクト
categor_acc = tf.keras.metrics.CategoricalAccuracy()
# 精度を測定するデータを設定
categor_acc.update_state(y_test, test_preds)
"""
'''
1. データセットの読み込みと前処理
'''
# tensorflowのインポート
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

                            
'''
2. モデルの定義  \\\class,定義のみ\\\
'''
#サブクラス継承
#子は親のメソッドを扱うことができる。子classMLPは親classtf.keras.Modelのメソッドを扱うことができる
#callはtf.keras.Modelからオーバーライド（上書き）したモノである
#callではパラメータで取得したデータをモデルに入力し、最終出力を戻り値として返す。学習、評価時に用いる
class MLP(tf.keras.Model):
    '''多層パーセプトロン
    
    Attributes:
      l1(Dense): 隠れ層
      l2(Dense): 出力層
    '''
    def __init__(self, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim4,output_dim):#hidden_dim5,hidden_dim6 ,
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
        # 隠れ層：活性化関数はReLU
        #self.fc1_5 = tf.keras.layers.Dense(hidden_dim5, activation='relu')
        # 隠れ層：活性化関数はReLU
        #self.fc1_6 = tf.keras.layers.Dense(hidden_dim6, activation='relu')
        # 出力層：活性化関数はシグモイド
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
        #x = self.fc1_5(x) #第5層の出力
        #x = self.fc1_6(x) #第6層の出力
        x = self.fc2(x) # 出力層の出力
        return x

@tf.function
def train_step(x, t):
    '''学習を1回行う この学習とはMNISTの場合だと60000枚の画像を全て読み込ませて1回ということになる
    
    Parameters: x(ndarray(float32)):訓練データ
                t(ndarray(float32)):正解ラベル　訓練ラベル
                
    Returns:
      ステップごとのクロスエントロピー誤差
    '''
    # 自動微分による勾配計算を記録するブロック
    # with.... 開始と終了を必要とする処理 安全にclassの処理を行ってくれるhttps://qiita.com/tchnkmr/items/c6b6480d236a51b91c17
    with tf.GradientTape() as tape:
        # 訓練モードをTrueに指定し、
        # モデルに入力して順伝搬の出力値を取得
        # 訓練モードが終了したら終わるのか？(withだから安全に)
        #modelは入力パラメータxをいれて出力yを出す。model=MLP()
        #model(inputs)でcall(inputs,training= ...)を呼び出すらしい https://runebook.dev/ja/docs/tensorflow/keras/model
        outputs = model(x, training=True)
        # 出力値と正解ラベルの誤差　(エントロピー)
        tmp_loss = loss_fn(t, outputs)
        #訓練モードのまま誤差を計算し、↓に流す
        
    # tapeに記録された操作を使用して誤差の勾配を計算        
    grads = tape.gradient(
        # 現在のステップの誤差
        tmp_loss,
        # バイアス、重みのリストを取得
        model.trainable_variables)
    # 勾配降下法の更新式を適用してバイアス、重みを更新
    #zip関数はリストをひとまとめにして取得することができる
    optimizer.apply_gradients(zip(grads,
                                  model.trainable_variables))
    
    # 損失をMeanオブジェクトに記録　train_loss = tf.keras.metrics.Mean()
    train_loss(tmp_loss)
    #クラスのインスタンス呼び出し以下同様
    # 精度をCategoricalAccuracyオブジェクトに記録
    train_accuracy(t, outputs)

@tf.function
def valid_step(val_x, val_y):
    # 訓練モードをFalseに指定し、
    # モデルに入力して順伝搬の出力値を取得
    pred = model(val_x, training = False)
    # 出力値と正解ラベルの誤差
    tmp_loss = loss_fn(val_y, pred)
    # 損失をMeanオブジェクトに記録
    val_loss(tmp_loss) #クラスのインスタンス
    # 精度をCategoricalAccuracyオブジェクトに記録
    val_accuracy(val_y, pred)

"---------------csv読み込み---------------"

'''
 3. 損失関数とオプティマイザーの生成
'''

# マルチクラス分類のクロスエントロピー誤差を求めるオブジェクト
#loss_fn(正解ラベル, 出力)
#loss_fn = tf.keras.losses.CategoricalCrossentropy()

#マルチラベル分類ではbinary_crossentropyを使うべし#############
loss_fn = tf.keras.losses.BinaryCrossentropy()

# 勾配降下アルゴリズムを使用するオプティマイザーを生成
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


'''
4. 勾配降下アルゴリズムによるパラメーターの更新処理を行うtrain_step()関数
'''
# 損失を記録するオブジェクトを生成
train_loss = tf.keras.metrics.Mean()
# カテゴリカルデータの精度を記録するオブジェクトを生成 ###############マルチラベル分類では.CategoricalAccuracy()はどうなるのか
#分類精度を記録
train_accuracy = tf.keras.metrics.BinaryAccuracy()

'''
5. 検証を行うvalid_step()関数    validation : 検証
'''
# 損失を記録するオブジェクトを生成
val_loss = tf.keras.metrics.Mean()
# カテゴリカルデータの精度を記録するオブジェクトを生成 ##############同様
val_accuracy = tf.keras.metrics.BinaryAccuracy()

# エポック数
epochs = 30
#epochs = 5
# ミニバッチのサイズ
batch_size = 64


# MLP(隠れ層1のユニット数、隠れ層2のユニット数、出力層のユニット数)...モデルを生成 (セル2より)
model = MLP(6000, 6000, 6000, 6000, 995)#10, 100, 

# 損失と精度の履歴(history)を保存するためのdictオブジェクト
#dict(辞書型)リスト
#user_info = {'user_name': 'taro', 'last_name': 'Yamada'}
#print user_info
#{'last_name': 'Yamada', 'user_name': 'taro'}
#辞書からの値の取り出し
#print user_info['user_name']
#'taro' 
history = {'loss':[],'accuracy':[], 'val_loss':[], 'val_accuracy':[]}

# 早期終了の判定を行うオブジェクトを生成 (セル6より)
#ers = EarlyStopping(patience=10, # 監視対象回数
                        #verbose=1)  # 早期終了時にメッセージを出力


filenum = 100  #314 #何個のファイルを読み取るか   何個のインデックスを獲得するか(R_0,...R_filenaum)
fileset = 10      #何ファイルずつセットにしてarray化するか

# クラスの数 ####正解の数######
class_num = 995 #ちょっと不安だから数確認しとこうね
"相関行列R"    
#訓練データ 
for epoch in range(epochs):
    fileplus = 0
    #testfileplus = 0
    # 学習するたびに、記録された値をリセット
    train_loss.reset_states()     # 訓練時における損失の累計　　　# 損失をMeanオブジェクトに記録　train_loss = tf.keras.metrics.Mean()
    train_accuracy.reset_states() # 訓練時における精度の累計
    val_loss.reset_states()       # 検証時における損失の累計
    val_accuracy.reset_states()   # 検証時における精度の累計
    #損失と精度をリセットするだけ。重みはそのまま
    for xtrain_set in tqdm(range(int(filenum / fileset))): #(filenum:314)/(fileset:2)=157
        R_train = []
        d_train = []
        for ll in range(fileset):
            R_data = np.loadtxt(r'/home/user_05/program/R_d10per_random_60/R_10_'+str(ll+fileplus)+'.csv',delimiter=",")#D:/Program/R_d5per_random/R_d5per
            d_data = np.loadtxt(r'/home/user_05/program/d_10per_60_01/d_10per_60_01_'+str(ll+fileplus)+'.csv',delimiter=",")
            #shape...(98596, 7)
            R_train.extend(R_data)
            d_train.extend(d_data)
            #print("train_data読み込み完了"+str(xtrain_set))
        #print("data読み込み完了"+str(xtrain_set))
        
        fileplus += int(fileset)
        
        x_train = R_train
        x_train = np.array(x_train) #/ 2.1
        y_train = d_train
        y_train = np.array(y_train)

        #print(x_train)
        #print(x_train.shape)
        #print("y_train1",y_train1)
        #print("y_train1.shape",y_train1.shape)
        #print("x_test",x_test)
        #print("y_test",y_test)
        


        # 訓練データの正解ラベルをOne-Hot表現に変換    #####複数正解となると一つだけ1にするOne-Hotではないのではないか
        # One-Hot表現
        #y_trainrows, y_traincolumns = y_train1.shape #行数、列数を取得
        #y_train = np.zeros((y_trainrows,class_num),dtype='float32')
        #print(y_train.shape)
        #for k in tqdm(range(y_trainrows)):
        #    y_train[k] = [1 if aa in y_train1[k] else 0 for aa in range(class_num)]
        #print("y_train",y_train1[1000])
        #print("y_train",y_train)
        #print("y_train.shape",y_train.shape)
        #print("y_train[1000]_",y_train[1000])

        

        '''
        7.訓練データと検証データの用意（x_train, y_trainを分割する行程）
        '''
        

        # 訓練データと検証データに8：2の割合で分割  \は行継続文字
        # これはランダムに分割されるのかな？　→　ランダムになる
        tr_x, val_x, tr_y, val_y = \
            train_test_split(x_train, y_train, test_size=0.2)
        #print(tr_x.shape) #(78876, 7) 1fileだけ読み込みの場合
        #print(val_x.shape)#(19720, 7) 1fileだけ読み込みの場合
        #print(tr_y.shape)#(78876, 1000) 1fileだけ読み込みの場合
        #print(val_y.shape)#(19720, 1000) 1fileだけ読み込みの場合
        
        
        #セル内のコードの実行時間を測定するためのもの
        '''
        8.モデルを生成して学習する
        '''

        # 訓練データのステップ数　ダブルスラッシュは切り捨て除算 5 // 3 = 1
        tr_steps = tr_x.shape[0] // batch_size
        # 検証データのステップ数
        val_steps = val_x.shape[0] // batch_size

        #epochの元の場所
        
        #訓練データと正解ラベルをシャッフル
        #tr_x(48000, 784)
        "ここをランダムなR_list() ←　入力パラメータからランダムな要素を取ってくるみたいな感じにしたい。何個取るか…ミニバッチのサイズ分"
        x_, y_ = shuffle(tr_x, tr_y, )
        ""
        #l0 = model.layers[0]  #各レイヤーの重みの数確認、l0とかの定義はどこでするべきか確認
        #l1 = model.layers[1]
        #l2 = model.layers[2]
        #print("l0weights_bef",l0.get_weights())
        #print("l1weights_bef",l1.get_weights())
        #print("l2weights_bef",l2.get_weights())
        # 1ステップにおける訓練用ミニバッチを使用した学習
        for step in range(tr_steps):
            start = step * batch_size # ミニバッチの先頭インデックス
            end = start + batch_size  # ミニバッチの末尾のインデックス
            # ミニバッチでバイアス、重みを更新して誤差を取得
            train_step(x_[start:end], y_[start:end])#(セル4train_step関数より)
        # 勾配降下アルゴリズムによるパラメーターの更新処理を行うtrain_step()関数

        # 1ステップにおける検証用ミニバッチを使用した評価
        for step in range(val_steps):
            start = step * batch_size # ミニバッチの先頭インデックス
            end = start + batch_size  # ミニバッチの末尾のインデックス
            # ミニバッチでバイアス、重みを更新して誤差を取得
            valid_step(val_x[start:end], val_y[start:end])#(セル5valid_step関数より)

    avg_train_loss = train_loss.result()    # 訓練時の平均損失値を取得 上記train_step関数でtr_steps分のloss取得済み。
    avg_train_acc = train_accuracy.result() # 訓練時の平均正解率を取得
    avg_val_loss = val_loss.result()     # 検証時の平均損失値を取得
    avg_val_acc = val_accuracy.result()  # 検証時の平均正解率を取得

    # 損失の履歴を保存する
    history['loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    # 精度の履歴を保存する
    history['accuracy'].append(avg_train_acc)
    history['val_accuracy'].append(avg_val_acc)
        
    #最初のファイルセットから重みを決定し、それを保存、引継ぎをしたい。
    #model.save_weights("weightt.h5", save_format="h5")
    #model.load_weights('weightt.h5')
    #l0 = model.layers[0]  #各レイヤーの重みの数確認、l0とかの定義はどこでするべきか確認
    #l1 = model.layers[1]
    #l2 = model.layers[2]
    #print("l0weights_after",l0.get_weights())
    #print("l1weights_after",l1.get_weights())
    #print("l2weights_after",l2.get_weights())
    
    # 1エポックごとに結果を出力
    if (epoch + 1) % 1 == 0:
        print(
            'epoch({}) train_loss: {:.4} train_acc: {:.4} val_loss: {:.4} val_acc: {:.4}'.format(
                epoch+1,
                avg_train_loss,     # 現在の損失を出力
                avg_train_acc,      # 現在の精度を出力
                avg_val_loss, # 現在の損失を出力
                avg_val_acc   # 現在の精度を出力
    ))
        
        # 検証データの損失をEarlyStoppingオブジェクトに渡して早期終了を判定
        #if ers(val_loss.result()): #ers = EarlyStopping(patience=, verbose=)  # 監視対象回数# 早期終了時にメッセージを出力
            # 監視対象のエポックで損失が改善されなければ学習を終了
            #break
#モデルの保存
model.save('model_save_datanum10per_6000node_60') #モデルと重みの保存
#重みの保存
model.save_weights("weight_datanum10per_6000node_60v2.h5", save_format="h5")    
# モデルの概要を出力
model.summary()
'''
9. 損失の推移をグラフにする
'''
import matplotlib.pyplot as plt
#%matplotlib inline

# 訓練データの損失
plt.plot(history['loss'],
         marker='.',
         label='loss (Training)')
# 検証データの損失
plt.plot(history['val_loss'],
         marker='.',
         label='loss (Validation)') #validation : 検証
plt.legend(loc='best') # 凡例最適な位置にを出力
plt.grid()             # グリッドを表示
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

'''
10. 精度の推移をグラフにする
'''
# 訓練データの精度
plt.plot(history['accuracy'],
         marker='.',
         label='accuracy (Training)')
# 検証データの精度
plt.plot(history['val_accuracy'],
         marker='.',
         label='accuracy (Validation)')
plt.legend(loc='best') # 凡例最適な位置にを出力
plt.grid()             # グリッドを表示
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

'train only'
# 訓練データの損失
plt.plot(history['loss'],
         marker='.',
         label='loss (Training)')
plt.legend(loc='best') # 凡例最適な位置にを出力
plt.grid()             # グリッドを表示
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
'train only'
plt.plot(history['accuracy'],
         marker='.',
         label='accuracy (Training)')

plt.legend(loc='best') # 凡例最適な位置にを出力
plt.grid()             # グリッドを表示
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
'''
11. テストデータによるモデルの評価
'''
testfilenum = 100
testfileset = 4
# 損失を記録するオブジェクトを生成
test_loss = tf.keras.metrics.Mean()
#分類精度を記録
test_acc = tf.keras.metrics.BinaryAccuracy()

history_test = {'loss_test':[],'accuracy_test':[]}

for test in range(int(testfilenum / testfileset)):
    fileplus = 0
    #testfileplus = 0
    # 学習するたびに、記録された値をリセット
    test_loss.reset_states()     # 訓練時における損失の累計　　　# 損失をMeanオブジェクトに記録　train_loss = tf.keras.metrics.Mean()
    test_acc.reset_states() # 訓練時における精度の累計
    
    #損失と精度をリセットするだけ。重みはそのまま
    R_test = []
    d_test = []
    for ll in tqdm(range(testfileset)):
        R_testdata = np.loadtxt(r'/home/user_05/program/R_d10per_random/R_10_' + str(ll + fileplus) + '.csv',delimiter=",")
        d_testdata = np.loadtxt(r'/home/user_05/program/d_10per_60/d_10per_test/d_10_' + str(ll + fileplus) + '.csv', delimiter=",")
        R_test.extend(R_testdata)
        d_test.extend(d_testdata)
    fileplus += int(testfileset)
    x_test = R_test
    x_test = np.array(x_test) #/  2.1 #正規化
    y_test = d_test
    y_test = np.array(y_test)
    # テストデータの予測値を取得
    test_preds = model(x_test)   #メソッド宣言した後に継承みたいな感じでかっこに入れるのなんだ？
    # カテゴリカルデータの精度を取得するオブジェクト
    categor_acc = tf.keras.metrics.BinaryAccuracy()
    # 精度を測定するデータを設定
    categor_acc.update_state(y_test, test_preds)
    # テストデータの精度を取得 #numpy形式に変換
    #どのような値になればいいんだ小さくなればいい？
    test_acc = categor_acc.result().numpy()
    # テストデータの損失を取得 
    test_loss = loss_fn(y_test, test_preds)

    avg_test_loss = test_loss.result()    # 訓練時の平均損失値を取得 上記train_step関数でtr_steps分のloss取得済み。
    avg_test_acc = test_acc.result() # 訓練時の平均正解率を取得

    # 損失の履歴を保存する
    history_test['loss_test'].append(avg_test_loss)
    history_test['accuracy_test'].append(avg_test_acc)

print('test_loss: {:.4f}, test_acc: {:.4f}'.format(
    avg_test_loss,
    avg_test_acc
))