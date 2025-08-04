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
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0].name}")
        
        # メモリの成長を許可（推奨設定）
        tf.config.experimental.set_memory_growth(gpus[0], True)
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

# 全体のコード内容は既存のプログラムと同じです。
# 修正箇所を強調します。

# 修正済みの予測精度計算関数
def calculate_prediction_accuracy(true_labels, predictions):
    """
    モデルの予測が正解データに含まれるかを確認。
    true_labels: 正解ラベル（ワンホットまたは複数の正解インデックスを含む形式）。
    predictions: モデル出力（確率値）。
    """
    # 出力値の最大インデックスを取得
    top_indices = tf.argmax(predictions, axis=1)  # 各サンプルで最大値のインデックスを取得

    # 各行で、最大インデックスが正解データに含まれるかを確認
    def is_match(args):
        top_index, true_row = args
        return tf.reduce_any(tf.equal(top_index, tf.where(true_row > 0)[:, 0]))

    matches = tf.map_fn(
        is_match,
        (top_indices, true_labels),
        fn_output_signature=tf.bool
    )

    # 精度を計算
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
    return accuracy

# 訓練ステップ関数（修正済み）
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)           # 勾配計算
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # パラメータ更新
    train_loss(loss)
    train_accuracy(y, predictions)
    prediction_accuracy = calculate_prediction_accuracy(y, predictions)
    train_prediction_accuracy.update_state(prediction_accuracy)

# テストステップ関数（修正済み）
@tf.function
def test_step(x, y):
    predictions = model(x, training=False)
    loss = loss_fn(y, predictions)
    val_loss(loss)
    val_accuracy(y, predictions)
    prediction_accuracy = calculate_prediction_accuracy(y, predictions)
    val_prediction_accuracy.update_state(prediction_accuracy)


#3. 損失関数とオプティマイザーの生成

# 損失関数と最適化アルゴリズム
loss_fn = tf.keras.losses.BinaryCrossentropy()
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

# 分類精度を記録するオブジェクトを追加
train_prediction_accuracy = tf.keras.metrics.Mean(name="train_prediction_accuracy")
val_prediction_accuracy = tf.keras.metrics.Mean(name="val_prediction_accuracy")


# モデル作成: 隠れ層を3層 (128, 256, 128ユニット) に設定
hidden_units_list = [3000, 3000, 3000, 3000]    
num_classes = 995  # 出力ラベル数
model = MLP(hidden_units_list, num_classes)

# 訓練ループ
epochs = 30
batch_size = 64

history = {'loss':[],'accuracy':[], 'val_loss':[], 'val_accuracy':[],'train_prediction_accuracy': [],
    'val_prediction_accuracy': []}

filenum = 100  #314 #何個のファイルを読み取るか   何個のインデックスを獲得するか(R_0,...R_filenaum)
fileset = 10      #何ファイルずつセットにしてarray化するか

for epoch in range(epochs):
    fileplus = 0
    train_loss.reset_states()     # 訓練時における損失の累計　　　# 損失をMeanオブジェクトに記録　train_loss = tf.keras.metrics.Mean()
    train_accuracy.reset_states() # 訓練時における精度の累計
    train_prediction_accuracy.reset_states()  # 訓練時における予測精度の累計
    val_loss.reset_states()       # 検証時における損失の累計
    val_accuracy.reset_states()   # 検証時における精度の累計
    val_prediction_accuracy.reset_states()  # 検証時における予測精度の累計
    for xtrain_set in tqdm(range(int(filenum / fileset))): #(filenum:314)/(fileset:2)=157
        R_train = []
        d_train = []
        # バッチ内のファイルを読み込み
        for ll in range(fileset):
            # 入力データとラベルデータを読み込む
            input_data = np.loadtxt("/home/user_05/program/R_d200per_random_90/R_200_"+str(ll+fileplus)+'.csv', delimiter=",")  # 入力データ (30000, 7)
            label_data = np.loadtxt("/home/user_05/program/d_200per_90_01/d_200per_01_"+str(ll+fileplus)+'.csv',delimiter=",")  # 教師データ (30000, 995)
            
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
    
        # 8.モデルを生成して学習する

        # 訓練データのステップ数　ダブルスラッシュは切り捨て除算 5 // 3 = 1
        tr_steps = tr_x.shape[0] // batch_size
        # 検証データのステップ数
        val_steps = val_x.shape[0] // batch_size


        x_, y_ = shuffle(tr_x, tr_y, )
    
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
           test_step(val_x[start:end], val_y[start:end])#(セル5valid_step関数より)

    avg_train_loss = train_loss.result()    # 訓練時の平均損失値を取得 上記train_step関数でtr_steps分のloss取得済み。
    avg_train_acc = train_accuracy.result() # 訓練時の平均正解率を取得
    avg_train_pred_acc = train_prediction_accuracy.result()
    avg_val_loss = val_loss.result()     # 検証時の平均損失値を取得
    avg_val_acc = val_accuracy.result()  # 検証時の平均正解率を取得
    avg_val_pred_acc = val_prediction_accuracy.result()

    # 損失の履歴を保存する
    history['loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    # 精度の履歴を保存する
    history['accuracy'].append(avg_train_acc)
    history['val_accuracy'].append(avg_val_acc)
    # 予測精度の履歴を保存する
    history['train_prediction_accuracy'].append(avg_train_pred_acc)
    history['val_prediction_accuracy'].append(avg_val_pred_acc)

    # 結果を出力
    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}, "
          f"Train Pred Accuracy: {avg_train_pred_acc:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Val Accuracy: {avg_val_acc:.4f}, Val Pred Accuracy: {avg_val_pred_acc:.4f}")
    
#モデルの保存
model.save('/home/user_05/program/model_save_200per_3000node_90') #モデルと重みの保存
#重みの保存
model.save_weights("/home/user_05/program/weight_datanum200per_3000node_90.h5", save_format="h5")    
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
fig.savefig('/home/user_05/d_loss_TandA/d200_loss_TandA_3000node_90.png')

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
fig2.savefig('/home/user_05/d_accuracy_TandA/d200_accuracy_TandA_3000node_90.png')

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
fig3.savefig('/home/user_05/d_loss_T/d200_loss_3000node_90.png')

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
fig4.savefig('/home/user_05/d_accuracy_T/d200_accuracy_3000node_90.png')

# 訓練データの予測精度
fig5 = plt.figure()
ax5 = fig5.add_subplot(1,1,1)
ax5.plot(history['train_prediction_accuracy'],
         marker='.',
         label='prediction_accuracy (Training)')
# 検証データの損失
ax5.plot(history['val_prediction_accuracy'],
         marker='.',
         label='prediction_accuracy (Validation)') #validation : 検証
ax5.legend(loc='best') # 凡例最適な位置にを出力
ax5.grid()             # グリッドを表示
ax5.set_ylabel('prediction_accuracy')
ax5.set_xlabel('epoch')
#plt.show()
fig5.savefig('/home/user_05/d_prediction_accuracy_TandA/d200_prediction_TandA_3000node_90.png')


