import tensorflow as tf
from tensorflow import keras
# from keras import layers
# import keras
# import keras.backend as K
import numpy as np
import decode, encode
import generate_data

INPUT_NODE = 2  # 输入节点
OUTPUT_NODE = 16  # 输出节点

LEARNING_RATE_BASE = 0.01 # 模型基础学习速率
LEARNING_RATE_DECAY = 0.99  # 学习衰减速度
BATCH_SIZE = 100  # 一批数据量
TRAIN_NUM = 2000  # 数据总量
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减
TRAINING_STEPS = 1000  # 训练多少次

SNR = -2   # 信噪比

E_x = 10 ** (0.1*SNR)  # 信号能量


# 生成数据
Y = generate_data.generate_bit([TRAIN_NUM, OUTPUT_NODE])  # 让数据的0和1的概率都相同

X = encode.encode2d(Y).reshape(-1, INPUT_NODE)  # TRAIN_NUM , OUTPUT_NODE / 2 , 2

X_onehot = encode.encode2d_onehot(Y)


# 验证数据
Y_vaildate = generate_data.generate_bit([TRAIN_NUM, OUTPUT_NODE])

X_validate = encode.encode2d(Y).reshape(-1, INPUT_NODE)  # TRAIN_NUM , OUTPUT_NODE / 2 , 2

X_validate_onehot = encode.encode2d_onehot(Y)

# 加噪声
X_shape = X.shape
noise = np.random.randn(X_shape[0], X_shape[1])  # sigma * r + mu
X = X * E_x + noise

# 定义整个模型的x和y
x = tf.placeholder(tf.float32, [None, INPUT_NODE, 2], name='x_input')

y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

input_dnn = tf.contrib.layers.batch_norm(x, is_training=True)  # norm

model_dnn = keras.layers.Dense(16, activation='relu')(input_dnn)

model_dnn = keras.layers.Dense(32, activation='relu')(model_dnn)

model_dnn = keras.layers.Dense(16, activation='relu')(model_dnn)

output_dnn = keras.layers.Dense(16, activation='softmax')(model_dnn)

max_index = tf.argmax(output_dnn)


def judge_onehot(a):
    if a <= 3:
        return np.array([1., 0., 0., 0.0]).astype(np.float32)
    elif a <= 7:
        return np.array([0., 1., 0., 0.0]).astype(np.float32)
    elif a <= 11:
        return np.array([0., 0., 1., 0.0]).astype(np.float32)
    elif a <= 15:
        return np.array([0., 0., 0., 1.0]).astype(np.float32)


y = tf.py_func(judge_onehot, [max_index], tf.float32)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,
                                                        labels=y_)  # 自动one-hot编码
cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 平均交叉熵

global_step = tf.get_variable('global_step',  # 存储当前迭代的轮数
                              dtype=tf.int32,  # 整数
                              initializer=0,  # 初始化值
                              trainable=False)  # 不可训练

learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,  # 基础学习率
                                           global_step,  # 当前迭代轮数
                                           TRAIN_NUM / BATCH_SIZE,  # 迭代次数
                                           LEARNING_RATE_DECAY,  # 学习衰减速度
                                           staircase=False)  # 是否每步都改变速率

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_mean, global_step)

with tf.Session() as sess:

    tf.global_variables_initializer().run()  # 初始化
    for i in range(TRAINING_STEPS):
        # 设置批次
        start = (i * BATCH_SIZE) % TRAIN_NUM
        end = min(start + BATCH_SIZE, TRAIN_NUM)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        compute_loss = sess.run(cross_entropy_mean,
                                feed_dict={x: X[start:end], y_: Y[start:end]})

        validate_loss = sess.run(cross_entropy_mean,
                                 feed_dict={x: X_validate[start:end],
                                            y_: Y_vaildate[start:end]})

        # 输出
        if i % 100 == 0:
            print('训练了%d次,总损失%f,验证损失%f' % (i, compute_loss, validate_loss))

        if (i % 500 == 0) and (i != 0):
            print('模型预测结果:', sess.run(y, feed_dict={x: X[start:start + 1], y_: Y[start:start + 1]}))
            print('实际结果:', sess.run(y_, feed_dict={x: X[start:start + 1], y_: Y[start:start + 1]}))
            print('加上噪声:', sess.run(x, feed_dict={x: X[start:start + 1], y_: Y[start:start + 1]}))
            # print('去掉噪声:', sess.run(data_after_remove_voice, feed_dict={x: X[start:start + 1], y_: Y[start:start + 1]}))
            # print('批归一化:', sess.run(input_cnn, feed_dict={x: X[start:start+1], y_: Y[start:start+1]}))
        #     # print('DNN后:', sess.run(y, feed_dict={x: X[start:end], y_: Y[start:end]}))
        #     print('weight:', sess.run(weight, feed_dict={x: X[start:end], y_: Y[start:end]}))

sess.close()