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
    def call(self, inputs, training=None):
        """
        順伝播を定義
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)  # 隠れ層を順に適用
        return self.output_layer(x)  # 出力層

@tf.function
def compute_predicted_sinr(predictions, angles):
    # predictions.shape = [batch_size, ...]
    # angles.shape = [batch_size, angle_dimension]

    # データ型統一
    predictions = tf.cast(predictions, tf.float64)
    angles = tf.cast(angles, tf.float64)

    # TensorArray を使用してテンソルを格納
    batch_size = tf.shape(predictions)[0]
    sinr_array = tf.TensorArray(dtype=tf.float64, size=batch_size)

    # バッチごとに計算
    for i in tf.range(batch_size):
        prediction_row = predictions[i]
        angle_row = angles[i]

        # 予測された素子間隔を取得
        top_index = tf.argmax(prediction_row)  # 最大インデックス
        predicted_spacing = tf.gather(d_values, top_index)

        # SINR の計算
        cos_values_10 = tf.cos(2.0 * np.pi * predicted_spacing * (tf.sin(angle_row[1]) - tf.sin(angle_row[0]))) / wavelength
        cos_values_20 = tf.cos(2.0 * np.pi * predicted_spacing * (tf.sin(angle_row[2]) - tf.sin(angle_row[0]))) / wavelength
        result1 = 2.0 * Parray[0] * ((Parray[1] * (1.0 - cos_values_10) + Parray[2] * (1.0 - cos_values_20)) + Pz)

        result2 = 2.0 * Pz * pn

        cos_diff = tf.cos(2.0 * np.pi * predicted_spacing * (tf.sin(angle_row[1]) - tf.sin(angle_row[2])) / wavelength)
        result3 = 2.0 * Parray[1] * Parray[2] * (1.0 - cos_diff)

        sinr = result1 / (result2 + result3 + (Pz ** 2))
        sinr_db = 10.0 * tf.math.log(sinr) / tf.math.log(tf.cast(10.0, tf.float64))

        # TensorArray に格納
        sinr_array = sinr_array.write(i, sinr_db)

    # TensorArray からテンソルに変換
    predicted_sinr_list = sinr_array.stack()
    predicted_sinr_list = tf.reshape(predicted_sinr_list, [-1])  # 明示的に形状を設定
    return predicted_sinr_list


# カスタム損失関数の変更
@tf.function
def custom_loss_fn(y_true, predictions, angles, true_sinr, predicted_sinr_list):
    # true_sinr の形状を調整
    true_sinr = tf.reduce_mean(true_sinr, axis=1)  # 必要に応じて平均を取る

    # SINR の差分を計算
    sinr_difference = tf.abs(predicted_sinr_list - true_sinr)
    sinr_difference_normalized = tf.clip_by_value(sinr_difference, 0.0, 1.0)
    sinr_penalty = tf.reduce_mean(sinr_difference_normalized)
    
    # sinr_penalty を float32 にキャスト
    sinr_penalty = tf.cast(sinr_penalty, tf.float32)
    alpha = 0.1
    bce = tf.keras.losses.BinaryCrossentropy()
    original_loss = bce(y_true, predictions)
    total_loss = original_loss + alpha * sinr_penalty

    return total_loss


# 訓練ステップでの使用
@tf.function
def train_step(x, y, angles, true_sinr):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        predicted_sinr_list = compute_predicted_sinr(predictions, angles)  # SINR計算
        loss = custom_loss_fn(y, predictions, angles, true_sinr, predicted_sinr_list)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(y, predictions)


# テストステップ関数
@tf.function
def test_step(x, y, angles, true_sinr):
    predictions = model(x, training=False)
    loss = custom_loss_fn(y, predictions, true_sinr, angles, true_sinr)
    # 損失をMeanオブジェクトに記録
    val_loss(loss) #クラスのインスタンス
    # 精度をCategoricalAccuracyオブジェクトに記録
    val_accuracy(y, predictions)

#3. 損失関数とオプティマイザーの生成

# 損失関数と最適化アルゴリズム
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

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

# モデル作成: 隠れ層を3層 (128, 256, 128ユニット) に設定
hidden_units_list = [3000, 3000, 3000, 3000]    
num_classes = 995  # 出力ラベル数
model = MLP(hidden_units_list, num_classes)

#　波長
wavelength = 0.1
# 素子間隔の範囲を設定
d_values = tf.Variable(np.linspace(0.5 * wavelength, 100 * wavelength, 995), dtype=tf.float64)

# 干渉波の数
i= 2

Parray1 = [1/i for s in range(i)]   #干渉波合計電力
pn = sum(Parray1[n] for n in range(0,i)) #変数i
p0 = [1]
Parray = np.hstack((p0, Parray1))
#print("Parray=",Parray[2])
Pz = 0.01

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
        angles  = []
        sinr = []
        # バッチ内のファイルを読み込み
        for ll in range(fileset):
            # 入力データとラベルデータを読み込む
            input_data = np.loadtxt("/home/user_05/program/R_50per_random_90_ver2/R_50_"+str(ll+fileplus)+'.csv', delimiter=",")  # 入力データ (30000, 7)
            label_data = np.loadtxt("/home/user_05/program/d_50per_90_ver2_01/d_50per_01_"+str(ll+fileplus)+'.csv',delimiter=",")  # 教師データ (30000, 995)
            angle_data = np.loadtxt("/home/user_05/program/DoA_50per_theta_ver2/DoA_50_"+str(ll+fileplus)+'.csv',delimiter=",")   # 到来方向 (30000, 3)
            sinr_data = np.loadtxt("/home/user_05/program/SINR_50per/SINR_50_"+str(ll+fileplus)+'.csv',delimiter=",")   # 各正解データに対するsinrの平均 (30000, 50)
            R_train.extend(input_data)
            d_train.extend(label_data)
            angles.extend(angle_data)
            sinr.extend(sinr_data)

        fileplus += int(fileset)

        x_train = R_train
        x_train = np.array(x_train) #/ 2.1
        y_train = d_train
        y_train = np.array(y_train)

        angles = angles
        angles = np.array(angles)
        sinr   = sinr
        sinr   = np.array(sinr)
        sinr   = np.mean(sinr, axis=1)

        #print(f"Initial sinr shape: {np.array(sinr).shape}")


        #7.訓練データと検証データの用意（x_train, y_trainを分割する行程）
        #       
        # 訓練データと検証データに8：2の割合で分割  \は行継続文字
        # これはランダムに分割されるのかな？　→　ランダムになる
        tr_x, val_x, tr_y, val_y, tr_angles, val_angles, tr_sinr, val_sinr = \
        train_test_split(x_train, y_train, angles, sinr, test_size=0.2) 
        
        #print(f"tr_sinr shape: {tr_sinr.shape}, val_sinr shape: {val_sinr.shape}")
    
        # 8.モデルを生成して学習する

        # 訓練データのステップ数　ダブルスラッシュは切り捨て除算 5 // 3 = 1
        tr_steps = tr_x.shape[0] // batch_size
        # 検証データのステップ数
        val_steps = val_x.shape[0] // batch_size


        x_, y_, angles_, sinr_ = shuffle(tr_x, tr_y, tr_angles, tr_sinr, )


        # 1ステップにおける訓練用ミニバッチを使用した学習
        for step in range(tr_steps):
           start = step * batch_size # ミニバッチの先頭インデックス
           end = start + batch_size  # ミニバッチの末尾のインデックス
           # ミニバッチでバイアス、重みを更新して誤差を取得
           #print(f"sinr_[start:end] shape: {sinr_[start:end].shape}")
           train_step(x_[start:end], y_[start:end], angles_[start:end], sinr_[start:end])#(セル4train_step関数より)
        # 勾配降下アルゴリズムによるパラメーターの更新処理を行うtrain_step()関数

        # 1ステップにおける検証用ミニバッチを使用した評価
        for step in range(val_steps):
           start = step * batch_size # ミニバッチの先頭インデックス
           end = start + batch_size  # ミニバッチの末尾のインデックス
           # ミニバッチでバイアス、重みを更新して誤差を取得
           test_step(val_x[start:end], val_y[start:end], val_angles[start:end], val_sinr[start:end])#(セル5valid_step関数より)

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
model.save('/home/user_05/program/model_save_50per_3000node_losssinr') #モデルと重みの保存
#重みの保存
model.save_weights("/home/user_05/program/weight_datanum50per_3000node_losssinr.h5", save_format="h5")    
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
fig.savefig('d_loss_TandA_3000node_losssinr.png')

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
fig2.savefig('d_accuracy_TandA_3000node_losssinr.png')

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
fig3.savefig('d_loss_3000node_losssinr.png')

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
fig4.savefig('d_accuracy_3000node_losssinr.png')