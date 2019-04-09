import tensorflow as tf
from tensorflow import keras
# from keras import layers
# import keras
# import keras.backend as K
import numpy as np
import decode, encode
import generate_data
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

INPUT_NODE = 32 # 输入节点
OUTPUT_NODE = 64 # 输出节点

LEARNING_RATE_BASE = 0.01 # 模型基础学习速率
LEARNING_RATE_DECAY = 0.99  # 学习衰减速度
BATCH_SIZE = 100  # 一批数据量
TRAIN_NUM = 20000  # 数据总量
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减
TRAINING_STEPS = 1000  # 训练多少次

SNR = -2   # 信噪比

E_x = 10 ** (0.1*SNR)  # 信号能量


# 生成数据
# Y = np.random.randint(0,2,[TRAIN_NUM , OUTPUT_NODE]).astype('float32')
Y = generate_data.generate_bit([TRAIN_NUM, OUTPUT_NODE])  # 让数据的0和1的概率都相同

X = encode.encode2d(Y)  # TRAIN_NUM , OUTPUT_NODE / 2 , 2

# 验证数据测试集
Y_vaildate = generate_data.generate_bit([TRAIN_NUM, OUTPUT_NODE])

X_validate = encode.encode2d(Y)  # TRAIN_NUM , OUTPUT_NODE / 2 , 2

# 加噪声
noise = np.random.randn(TRAIN_NUM, INPUT_NODE, 2)  # sigma * r + mu
X = X * E_x + noise

# 定义整个模型的x和y
x = tf.placeholder(tf.float32, [None, INPUT_NODE, 2], name='x_input')

y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

noise_ = tf.placeholder(tf.float32, [None, INPUT_NODE, 2], name='noise')


input_cnn = tf.contrib.layers.batch_norm(tf.reshape(x, [-1, OUTPUT_NODE, 1]), is_training=True)  # reshape + norm

model_cnn = keras.layers.Conv1D(32, 4, activation='relu', input_shape=(None, OUTPUT_NODE, 1))(input_cnn)  # 第一层卷积层

model_cnn = keras.layers.MaxPool1D(2)(model_cnn)  # 第一层最大池化层

model_cnn = keras.layers.Conv1D(64, 4, activation='relu')(model_cnn)  # 第二层卷积层

model_cnn = keras.layers.GlobalAvgPool1D()(model_cnn)  # 全局平均池化

model_cnn = keras.layers.Dense(64)(model_cnn)  # 密集层

receive_data_after_cnn = tf.reshape(model_cnn, [-1, INPUT_NODE, 2])

# 移除噪声
data_after_remove_voice = tf.subtract(x, receive_data_after_cnn)

y = tf.py_func(decode.decode2d, [data_after_remove_voice], np.float32)

"""
model = keras.models.Model(inputs=input_, outputs=y)

adam = keras.optimizers.Adam(lr=0.01, epsilon=1e-8)

model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit({'x_input': X,
                    'y_input': Y},
                    epochs=3,
                    batch_size=100,
                    validation_data=(X_validate,Y_vaildate),
                    validation_steps=100)

#score = model.evaluate(X_validate, Y_vaildate, batch_size=100)

#print('training loss: %f, val loss:%f, val acc:%f'%(history['loss'], history['val_loss'], history['val_acc']))
"""

# 损失函数
# cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y,
#                                                         labels=y_)  # 自动one-hot编码
# cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 平均交叉熵
#
# loss = cross_entropy_mean

# MSE
loss = tf.reduce_mean(tf.square(receive_data_after_cnn - noise_))  # cnn输出的噪声和加性噪声的均方误差

# 优化器
# 定义当前迭代轮数的变量
global_step = tf.get_variable('global_step',  # 存储当前迭代的轮数
                              dtype=tf.int32,  # 整数
                              initializer=0,  # 初始化值
                              trainable=False)  # 不可训练
# 定义学习速率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,  # 基础学习率
                                           global_step,  # 当前迭代轮数
                                           TRAIN_NUM / BATCH_SIZE,  # 迭代次数
                                           LEARNING_RATE_DECAY,  # 学习衰减速度
                                           staircase=False)  # 是否每步都改变速率
# 误码率
ber = tf.reduce_mean(tf.square(y - y_))

loss = loss + ber

# 定义优化函数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

"""
# 滑动平均类
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
train_op = tf.group(train_step, variables_averages_op)
"""


with tf.Session() as sess:

    tf.global_variables_initializer().run()  # 初始化
    for i in range(TRAINING_STEPS):
        # 设置批次
        start = (i * BATCH_SIZE) % TRAIN_NUM
        end = min(start + BATCH_SIZE, TRAIN_NUM)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end], noise_: noise[start:end]})

        ber_loss = sess.run(ber,
                            feed_dict={x: X[start:end], y_: Y[start:end]})

        compute_loss = sess.run(loss,
                                feed_dict={x: X[start:end], y_: Y[start:end], noise_: noise[start:end]})

        validate_loss = sess.run(loss,
                                 feed_dict={x: X_validate[start:end],
                                            y_: Y_vaildate[start:end],
                                            noise_: noise[start:end]})

        # 输出
        if i % 20 == 0:
            print('训练了%d次,总损失%f,ber为%f,验证损失%f' % (i, compute_loss, ber_loss, validate_loss))

        if (i % 500 == 0) and (i != 0):
            print('模型预测结果:', sess.run(y, feed_dict={x: X[start:start + 1], y_: Y[start:start + 1]}))
            print('实际结果:', sess.run(y_, feed_dict={x: X[start:start + 1], y_: Y[start:start + 1]}))
            print('加上噪声:', sess.run(x, feed_dict={x: X[start:start + 1], y_: Y[start:start + 1]}))
            print('去掉噪声:', sess.run(data_after_remove_voice, feed_dict={x: X[start:start + 1], y_: Y[start:start + 1]}))
            print('批归一化:', sess.run(input_cnn, feed_dict={x: X[start:start+1], y_: Y[start:start+1]}))
        #     # print('DNN后:', sess.run(y, feed_dict={x: X[start:end], y_: Y[start:end]}))
        #     print('weight:', sess.run(weight, feed_dict={x: X[start:end], y_: Y[start:end]}))

sess.close()

