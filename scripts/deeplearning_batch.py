# データセットの読み込みと前処理
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import csv
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
'''
# データを読み込む関数
def load_data(input_file, label_file):
    """
    入力データとラベルデータを読み込む。
    input_file: 入力データ（相関行列）のファイルパス
    label_file: 教師データ（正解ラベル）のファイルパス
    """
    # 入力データと教師データをロード（例: NumPyのバイナリ形式）
    X = np.load(input_file)  # shape: (30000, 7)
    Y = np.load(label_file)  # shape: (30000, 995)
    return X, Y

# 入力データと教師データを読み込み
input_file = "input_data.npy"  # 相関行列データ
label_file = "labels.npy"      # 教師データ
X, Y = load_data(input_file, label_file)
'''

# モデルの定義
class MLP(tf.keras.Model):
    def __init__(self, hidden_units_list, output_units):
        """
        隠れ層を多層構造にしたニューラルネットワーク
        hidden_units_list: 隠れ層のユニット数をリストで指定
        output_units: 出力ユニット数 (クラス数)
        """
        super(MLP, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(units, activation='relu') 
            for units in hidden_units_list
        ]
        self.output_layer = tf.keras.layers.Dense(output_units, activation='sigmoid')  # マルチラベル
    
    @tf.function
    def call(self, inputs, training=False):
        """
        順伝播を定義
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)  # 隠れ層を順に適用
        return self.output_layer(x)  # 出力層
    

# モデル作成: 隠れ層を3層 (128, 256, 128ユニット) に設定
hidden_units_list = [2000, 2000, 2000, 2000]    
num_classes = 995  # 出力ラベル数
model = MLP(hidden_units_list, num_classes)

# 損失関数と最適化アルゴリズム
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 訓練ステップ関数
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)           # 勾配計算
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # パラメータ更新
    train_loss(loss)
    train_accuracy(y, predictions)

# テストステップ関数
@tf.function
def test_step(x, y):
    predictions = model(x, training=False)
    loss = loss_fn(y, predictions)
    # 損失をMeanオブジェクトに記録
    val_loss(loss) #クラスのインスタンス
    # 精度をCategoricalAccuracyオブジェクトに記録
    val_accuracy(y, predictions)

#4. 勾配降下アルゴリズムによるパラメーターの更新処理を行うtrain_step()関数

# 損失を記録するオブジェクトを生成
train_loss = tf.keras.metrics.Mean()
# カテゴリカルデータの精度を記録するオブジェクトを生成 ###############マルチラベル分類では.CategoricalAccuracy()はどうなるのか
#分類精度を記録
train_accuracy = tf.keras.metrics.BinaryAccuracy()


#5. 検証を行うvalid_step()関数    validation : 検証

# 損失を記録するオブジェクトを生成
val_loss = tf.keras.metrics.Mean()
# カテゴリカルデータの精度を記録するオブジェクトを生成 ##############同様
val_accuracy = tf.keras.metrics.BinaryAccuracy()

# 訓練ループ
epochs = 30
batch_size = 64

history = {'loss':[],'accuracy':[], 'val_loss':[], 'val_accuracy':[]}

filenum = 100  #314 #何個のファイルを読み取るか   何個のインデックスを獲得するか(R_0,...R_filenaum)
fileset = 10      #何ファイルずつセットにしてarray化するか

for epoch in range(epochs):
    fileplus = 0
    train_loss.reset_states()     # 訓練時における損失の累計　　　# 損失をMeanオブジェクトに記録　train_loss = tf.keras.metrics.Mean()
    train_accuracy.reset_states() # 訓練時における精度の累計
    val_loss.reset_states()       # 検証時における損失の累計
    val_accuracy.reset_states()   # 検証時における精度の累計
    for xtrain_set in tqdm(range(int(filenum / fileset))): #(filenum:314)/(fileset:2)=157
        R_train = []
        d_train = []
        # バッチ内のファイルを読み込み
        for ll in range(fileset):
            # 入力データとラベルデータを読み込む
            input_data = np.loadtxt("/home/user_05/program/DoA_50per_theta/DoA_50_"+str(ll+fileplus)+'.csv', delimiter=",")  # 入力データ (30000, 7)
            label_data = np.loadtxt("/home/user_05/program/d_50per_theta_01/d_50per_01_"+str(ll+fileplus)+'.csv',delimiter=",")  # 教師データ (30000, 995)
            
            R_train.extend(input_data)
            d_train.extend(label_data)

    fileplus += int(fileset)

    x_train = R_train
    x_train = np.array(x_train) #/ 2.1
    y_train = d_train
    y_train = np.array(y_train)

    #7.訓練データと検証データの用意（x_train, y_trainを分割する行程）
    #       
    # 訓練データと検証データに8：2の割合で分割  \は行継続文字
    # これはランダムに分割されるのかな？　→　ランダムになる
    tr_x, val_x, tr_y, val_y = \
        train_test_split(x_train, y_train, test_size=0.2) 
    
    
    train_step(tr_x, tr_y)#(セル4train_step関数より)
    # 勾配降下アルゴリズムによるパラメーターの更新処理を行うtrain_step()関数

   
    test_step(val_x, val_y)#(セル5valid_step関数より)

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
    
    # 結果を出力
    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
      f"Train Accuracy: {avg_train_acc:.4f}, Val Accuracy: {avg_val_acc:.4f}")

#モデルの保存
model.save('/home/user_05/program/model_save_50per_2000node_10ele_batch') #モデルと重みの保存
#重みの保存
model.save_weights("/home/user_05/program/weight_datanum50per_2000node_10ele_batch.h5", save_format="h5")    
# モデルの概要を出力
model.summary()

'''
9. 損失の推移をグラフにする
'''
import matplotlib.pyplot as plt
#%matplotlib inline

# 訓練データの損失
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(history['loss'],
         marker='.',
         label='loss (Training)')
# 検証データの損失
ax.plot(history['val_loss'],
         marker='.',
         label='loss (Validation)') #validation : 検証
ax.legend(loc='best') # 凡例最適な位置にを出力
ax.grid()             # グリッドを表示
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
#plt.show()
fig.savefig('d_loss_TandA_2000node_batch.png')

'''
10. 精度の推移をグラフにする
'''
# 訓練データの精度
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(history['accuracy'],
         marker='.',
         label='accuracy (Training)')
# 検証データの精度
ax2.plot(history['val_accuracy'],
         marker='.',
         label='accuracy (Validation)')
ax2.legend(loc='best') # 凡例最適な位置にを出力
ax2.grid()             # グリッドを表示
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
#plt.show()
fig2.savefig('d_accuracy_TandA_2000node_batch.png')

'train only'
# 訓練データの損失
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(history['loss'],
         marker='.',
         label='loss (Training)')
ax3.legend(loc='best') # 凡例最適な位置にを出力
ax3.grid()             # グリッドを表示
ax3.set_ylabel('loss')
ax3.set_xlabel('epoch')
#plt.show()
fig3.savefig('d_loss_2000node_batch.png')

'train only'
fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(history['accuracy'],
         marker='.',
         label='accuracy (Training)')

ax4.legend(loc='best') # 凡例最適な位置にを出力
ax4.grid()             # グリッドを表示
ax4.set_xlabel('epoch')
ax4.set_ylabel('accuracy')
#plt.show()
fig4.savefig('d_accuracy_2000node_batch.png')