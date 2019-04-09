import tensorflow as tf
import CNN_interface
import DNN_interface
import numpy as np
import encode, decode  # 编码
import time

INPUT_NODE = 32  # 输入节点
OUTPUT_NODE = 64  # 输出节点

sent_data_shape = None  # 发射机的输出维度
sent_dnn_DROP = 0.5  # 发射机的DNN的drop
sent_dnn_REGULARIZER_RATE = 1e-4  # 发射机的DNN的正则率

receive_data_after_cnn_shape = 64  # 接收机通过CNN网络之后的数据维度
receive_cnn_DROP = 0.5  # 接收机的CNN的drop，CNN内的一个密集层的drop
receive_cnn_REGULARIZER_RATE = 1e-4  # 接收机中CNN的密集层的正则率


receive_dnn_DROP = 0.2  # 接收机的DNN的drop
receive_dnn_REGULARIZER_RATE = 1e-4  # 接收机的DNN的正则率

LEARNING_RATE_BASE = 0.01 # 模型基础学习速率
LEARNING_RATE_DECAY = 0.99  # 学习衰减速度
BATCH_SIZE = 200  # 一批数据量
TRAIN_NUM = 20000  # 数据总量
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减
TRAINING_STEPS = 500  # 训练多少次

SNR = 5   # 信噪比

E_x = 10 ** (0.1*SNR)  # 信号能量


# 生成数据
Y = np.random.randint(0, 2, [TRAIN_NUM, OUTPUT_NODE]).astype('float32')

X = encode.encode2d(Y)  # TRAIN_NUM , OUTPUT_NODE / 2 , 2

# 验证数据
Y_vaildate = np.random.randint(0, 2, [TRAIN_NUM, OUTPUT_NODE]).astype('float32')

X_validate = encode.encode2d(Y)  # TRAIN_NUM , OUTPUT_NODE / 2 , 2


# 定义整个模型的x和y
x = tf.placeholder(tf.float32, [None, INPUT_NODE, 2], name='x_input')

y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

"""
# 发射机通过DNN输出的数据
# dnn_interface(input_tensor, output_shape, regularizer_rate=None, drop=None)
sent_data = DNN_interface.dnn_interface(input_tensor=x,
                                        output_shape=sent_data_shape,
                                        regularizer_rate=sent_dnn_REGULARIZER_RATE,
                                        drop=sent_dnn_DROP )
"""

# 过信道,加噪声
X = X * E_x + np.random.randn(TRAIN_NUM , INPUT_NODE, 2)  # sigma * r + mu


# 把虚部放在一起，实部放在一起,顺便归一化
def reshape_dim(a):
    temp = []
    a_mean = np.mean(a)
    a_std = np.std(a)

    for i in range(np.array(a).shape[0]):
        temp1 = []
        temp2 = []

        for j in a[i]:
            j_0 = (j[0] - a_mean) / a_std
            j_1 = (j[0] - a_mean) / a_std
            temp1.append(j_0)
            temp2.append(j_1)
        temp1.extend(temp2)
        temp.append(temp1)
    return np.array(temp).astype(np.float32)


x_cnn = tf.py_func(reshape_dim, [x], tf.float32)

# 批量归一化
# x_toone = tf.contrib.layers.batch_norm(x_cnn, is_training=True)

# # 接收机的CNN网络
# receive_data_after_cnn, weight = CNN_interface.cnn_interface(input_tensor=x_toone,
#                                                         output_shape=receive_data_after_cnn_shape,
#                                                         drop=receive_cnn_DROP,
#                                                         regularizer_rate=receive_cnn_REGULARIZER_RATE)

receive_data_after_dnn, weight = DNN_interface.dnn_interface(input_tensor=x_cnn,
                                                        output_shape=receive_data_after_cnn_shape,
                                                        drop=receive_cnn_DROP,
                                                        regularizer_rate=receive_cnn_REGULARIZER_RATE)

# 移除噪声
data_after_remove_voice = tf.subtract(x, receive_data_after_dnn)

# 用解码函数判断
y = tf.py_func(decode.decode2d, [data_after_remove_voice], tf.float32)


# # DNN后的判断函数
# def judge_dnn(data):
#
#     mat = [1.0,0.0]
#     data_after_judge = np.where(data>0, mat[0], mat[1])
#     return data_after_judge.astype(np.float32)
#
# y_judged = tf.py_func(judge_dnn, [data_judged], tf.float32)
"""
y_judged = tf.round(y)
y_judged = tf.where(tf.equal(y_judged,0), y_judged-1 , y_judged)
"""

# 损失函数
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y,
                                                        labels=y_)  # 自动one-hot编码
cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 平均交叉熵

loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  # 损失函数是交叉熵和正则化的和

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
# 均方误差
ber = tf.reduce_mean(tf.square(y  * np.cos(global_step) - y_ * np.sin(global_step)))

loss = loss + ber

# 定义优化函数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(ber, global_step)

"""
# 滑动平均类
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
train_op = tf.group(train_step, variables_averages_op)
"""

# 保存模型
saver = tf.train.Saver(max_to_keep=3)
min_loss = float('inf')

#使用tensorboard
#writer = tf.summary.FileWriter('/logs/log')

#writer.close()
#merged = tf.summary.merge_all()

#tf.summary.scalar('cross_entropy_mean_loss', cross_entropy_mean)  # tensorboard写入交叉熵误差
tf.summary.scalar('loss', loss)  # tensorboard写入总误差
tf.summary.scalar('BER', ber)

summary_writer = tf.summary.FileWriter('logs/log_'+'%s'%(time.strftime('%m_%d_%H_%M')))  #tensorboard保存目录，目录以时间命名


with tf.Session() as sess:
    # 初始化写日志的writer,并将当前Tensorflow计算图写入日志

    summary_writer.add_graph(sess.graph)  # 写入变量图

    tf.global_variables_initializer().run()  # 初始化
    for i in range(TRAINING_STEPS):
        # 设置批次
        start = (i * BATCH_SIZE) % TRAIN_NUM
        end = min(start+BATCH_SIZE, TRAIN_NUM)

        merged = tf.summary.merge_all()

        _, summary = sess.run([train_step, merged],
                                        feed_dict={x:X[start:end], y_:Y[start:end]})
        
        ber_loss = sess.run(ber,
                        feed_dict={x: X[start:end], y_: Y[start:end]})
        
        compute_loss = sess.run(loss,
                                feed_dict={x: X[start:end], y_: Y[start:end]})

        validate_loss = sess.run(loss,
                                feed_dict={x: X_validate[start:end], y_: Y_vaildate[start:end]})
        # # 保存模型
        # if compute_loss < min_loss:
        #     min_loss = compute_loss
        #     saver.save(sess, 'ckpt/min_loss_model.ckpt', global_step=i)

        # 写入tensorboard日志
        summary_writer.add_summary(summary,i)

        # 输出
        if i % 100 == 0:
            print('训练了%d次,总损失%f,ber为%f,验证损失%f'%(i,compute_loss,ber_loss,validate_loss))

        if (i % (TRAINING_STEPS-1) == 0) and (i != 0):
            print('模型预测结果:',sess.run(y , feed_dict={x: X[start:start+1], y_: Y[start:start+1]}))
            print('实际结果:',sess.run(y_ , feed_dict={x: X[start:start+1], y_: Y[start:start+1]}))
            print('加上噪声:', sess.run(x, feed_dict={x: X[start:start+1], y_: Y[start:start+1]}))
            print('去掉噪声:', sess.run(data_after_remove_voice, feed_dict={x: X[start:start+1], y_: Y[start:start+1]}))
            # print('批归一化:', sess.run(x_cnn, feed_dict={x: X[start:start+1], y_: Y[start:start+1]}))
            # print('DNN后:', sess.run(y, feed_dict={x: X[start:end], y_: Y[start:end]}))
            print('weight:', sess.run(weight, feed_dict={x: X[start:end], y_: Y[start:end]}))

sess.close()