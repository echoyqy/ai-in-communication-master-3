import tensorflow as tf
import numpy as np

NUM_CHANNELS = 1  # 输入通道数

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 2

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 2

# 第三层卷积层的尺寸和深度
CONV3_DEEP = 64
CONV3_SIZE = 2

# 全连接层的节点个数
FC_SIZE = 512

"""

layer1 conv1  deep=32,size=5*5,step=1,padding='SAME'  activate=relu

layer2 pool1  

layer3 conv2  deep=64,size=5*5,step=1,padding='SAME'  activate=relu

layer4 pool2

layer5 conv3  deep=64,size=5*5,step=1,padding='SAME'  activate=relu

layer6 pool3 and shape transport 

layer7 dense1  label=512

layer8 dense2  output layer

"""


# 定义卷积神经网络的向前传播过程
def cnn_interface(input_tensor, output_shape, drop=None, regularizer_rate=None):
    # 使用不同的命名空间来隔离不同层的变量，不必担心重名的问题
    # 卷积层使用全0填充

    #reshape
    input_tensor = tf.reshape(input_tensor, [-1,64,1])

    # 第一层卷积层
    with tf.variable_scope('layer1-conv1'):

        # 卷积层参数
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, 1, CONV1_DEEP],  # shape
            initializer=tf.truncated_normal_initializer(stddev=0.1))  # 截断正态分布的随机数，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        # 卷积
        conv1 = tf.nn.conv1d(input_tensor, conv1_weights, stride=2,
                             padding='SAME')  # 边长为3，深度32的过滤器，过滤移动步长2，全0填充
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))  # 添加偏执和激活函数

    #
    # # 第一层池化层
    # with tf.name_scope("layer2-pool1"):
    #     pool1 = tf.layers.max_pooling1d(relu1, pool_size=2, strides=2,
    #                            padding="SAME")  # 池化层，最大池化，降维一倍，过滤器边长为2，移动步长为2

    # 第二层卷积层
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))  # 深度为64，即64个卷积核

        conv2 = tf.nn.conv1d(relu1, conv2_weights, stride=2, padding='SAME')  # 与上一层连接，第二层卷积层
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # # 第二层池化层
    # with tf.name_scope("layer4-pool2"):
    #     pool2 = tf.layers.max_pooling1d(relu2, pool_size=2, strides=2,
    #                            padding="SAME")  # 池化层，最大池化，降维一倍，过滤器边长为2，移动步长为2

    # # 第三层卷积层
    # with tf.variable_scope("layer5-conv3"):
    #     conv3_weights = tf.get_variable(
    #         "weight", [ CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
    #         initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))  # 深度为64，即64个卷积核
    #
    #     conv3 = tf.nn.conv1d(relu2, conv3_weights, stride=2, padding='SAME')  # 与上一层连接，第二层卷积层
    #     relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    # 第三层池化层
    with tf.name_scope("layer6-pool3"):
        pool3 = tf.layers.max_pooling1d(relu2, pool_size=2, strides=2,
                               padding="SAME")  # 池化层，最大池化，降维一倍，过滤器边长为2，移动步长为2

        
        # 获取池化层的shape为一个List
        pool_shape = pool3.get_shape().as_list()

        # flatten
        nodes = pool_shape[1] * pool_shape[2]

        # 将此层的输出变成一个batch的向量
        reshaped = tf.reshape(pool3, [-1, nodes])
        
    # # 第一层密集层
    # with tf.variable_scope('layer7-fc1'):
    #     fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
    #                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     if regularizer_rate != None:
    #         tf.add_to_collection('losses',
    #                              tf.contrib.layers.l2_regularizer(regularizer_rate)(fc1_weights))  # 添加L2正则化
    #
    #     fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
    #
    #     fc1 = tf.matmul(reshaped, fc1_weights) + fc1_biases  # tanh激活函数


    # 输出层
    with tf.variable_scope('layer8-fc2'):
        fc2_weights = tf.get_variable("weight", [nodes, output_shape],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer_rate != None:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(fc2_weights))
        fc2_biases = tf.get_variable("bias", [output_shape], initializer=tf.constant_initializer(0.1))
        run = tf.matmul(reshaped, fc2_weights) + fc2_biases

    run = tf.reshape(run, [-1, int(output_shape / 2), 2])

    return run, fc2_weights


def cnn_interface_2d(input_tensor, output_shape, drop=None, regularizer_rate=None):

    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))  #卷积核

        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding="SAME")
        # ksize=[batch,height,width,channels]


    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [-1, nodes])

        # 第一层密集层
    with tf.variable_scope('layer7-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer_rate != None:
            tf.add_to_collection('losses',
                                     tf.contrib.layers.l2_regularizer(regularizer_rate)(fc1_weights))  # 添加L2正则化

        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.tanh(tf.matmul(reshaped, fc1_weights) + fc1_biases)  # tanh激活函数
        if drop != None:
                fc1 = tf.nn.dropout(fc1, drop)

        # 输出层
    with tf.variable_scope('layer8-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, output_shape],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer_rate != None:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(fc2_weights))
        fc2_biases = tf.get_variable("bias", [output_shape], initializer=tf.constant_initializer(0.1))
        run = tf.matmul(fc1, fc2_weights) + fc2_biases

    run = tf.reshape(run, [-1, int(output_shape / 2), 2])

    return run
