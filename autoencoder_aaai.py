import tensorflow as tf  
import numpy as np   
import data_helper
from tensorflow.python.framework import ops
def autoencoder(N,U_side,n_feature,batch_size,n_input,n_input1,num_examples,training_epochs,learning_rate):

    p = 0.3
    learning_rate = 0.001 
    training_epochs =20
    #batch_size = 151  
    display_step = 1  
    examples_to_show = 10  
    #n_input = 3952  
    #n_input1 = 29 
    #num_examples = 6040
   
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_input1])  
    X_new = tf.placeholder("float", [None, n_input])
    Y_new = tf.placeholder("float", [None, n_input1])   
    keep_prob = tf.placeholder(tf.float32)
    # 用字典的方式存储各隐藏层的参数  
    n_hidden_1 = 256 # 第一编码层神经元个数  
    n_hidden_2 = 100 # 第二编码层神经元个数  
    n_hidden_3 = 1024 # 第一编码层神经元个数  
    n_hidden_4 = 200
    # 权重和偏置的变化在编码层和解码层顺序是相逆的  
    # 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数  
    weights = {  
        'W1': tf.Variable(tf.random_uniform(([n_input, n_hidden_1]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
        'V1': tf.Variable(tf.random_uniform(([n_input1, n_hidden_1]),minval=-1.0,maxval=1.0,dtype=tf.float32)), 
        'W2': tf.Variable(tf.random_uniform(([n_hidden_1, n_hidden_2]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
        'V2': tf.Variable(tf.random_uniform(([n_input1, n_hidden_2]),minval=-1.0,maxval=1.0,dtype=tf.float32)),
        'W3': tf.Variable(tf.random_uniform(([n_hidden_2, n_hidden_3]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
        'V3': tf.Variable(tf.random_uniform(([n_input1, n_hidden_3]),minval=-1.0,maxval=1.0,dtype=tf.float32)), 
        'W4': tf.Variable(tf.random_uniform(([n_hidden_3, n_hidden_4]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
        'V4': tf.Variable(tf.random_uniform(([n_input1, n_hidden_4]),minval=-1.0,maxval=1.0,dtype=tf.float32)),

        'decoder_h1': tf.Variable(tf.random_uniform(([n_hidden_4, n_hidden_3]),minval=-1.0,maxval=1.0,dtype=tf.float32)), 
        'decoder_h11': tf.Variable(tf.random_uniform(([n_hidden_3, n_hidden_2]),minval=-1.0,maxval=1.0,dtype=tf.float32)),
        'decoder_h12': tf.Variable(tf.random_uniform(([n_hidden_2, n_hidden_1]),minval=-1.0,maxval=1.0,dtype=tf.float32)),
        'decoder_h4': tf.Variable(tf.random_uniform(([n_input1, n_hidden_3]),minval=-1.0,maxval=1.0,dtype=tf.float32)), 
        'decoder_h41': tf.Variable(tf.random_uniform(([n_input1, n_hidden_2]),minval=-1.0,maxval=1.0,dtype=tf.float32)), 
        'decoder_h42': tf.Variable(tf.random_uniform(([n_input1, n_hidden_1]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
        'decoder_h2': tf.Variable(tf.random_uniform(([n_hidden_1, n_input]),minval=-1.0,maxval=1.0,dtype=tf.float32)), 
        'decoder_h3': tf.Variable(tf.random_uniform(([n_hidden_1, n_input1]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
    }  
    biases = {  
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), 
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),  
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])), 
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),  
        'decoder_b11': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b111': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),  
        'decoder_b3': tf.Variable(tf.random_normal([n_input1])),
    }  

    # def selu(x):
    #     with ops.name_scope('elu') as scope:
    #         alpha = 1.6732632423543772848170429916717
    #         scale = 1.0507009873554804934193349852946
    #         return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
    def linear(input_,input_y,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        shape1 = input_y.get_shape().as_list()
        #print(shape)
        #print(ouput_size)
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        input_size1 = shape1[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix3344",[ouput_size,input_size],dtype=input_.dtype)
            matrix1 = tf.get_variable("Matrix55555",[ouput_size,input_size1],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias",[ouput_size],dtype=input_.dtype)
            Wx_plus_b = tf.matmul(input_,tf.transpose(matrix))+tf.matmul(input_y,tf.transpose(matrix1))+bias_term
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b ,matrix,matrix1
    def linear1(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix1",[ouput_size,input_size],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias1",[ouput_size],dtype=input_.dtype)
            Wx_plus_b=tf.matmul(input_,tf.transpose(matrix))+bias_term 
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b,matrix 
    def linear2(input_,input_y,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        shape1 = input_y.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        input_size1 = shape1[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix2",[ouput_size,input_size],dtype=input_.dtype)
            matrix1 = tf.get_variable("Matrix3776",[ouput_size,input_size1],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias2",[ouput_size],dtype=input_.dtype)
            Wx_plus_b=tf.matmul(input_,tf.transpose(matrix))+tf.matmul(input_y,tf.transpose(matrix1))+bias_term 
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return  Wx_plus_b,matrix,matrix1

    def linear3(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix3",[ouput_size,input_size],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias3",[ouput_size],dtype=input_.dtype)
            Wx_plus_b=tf.matmul(input_,tf.transpose(matrix))+bias_term
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b,matrix 
    def linear4(input_,input_y,ouput_size,scope=None):
        # shape = input_.get_shape().as_list()
        # #print(shape)
        # #print(ouput_size)
        # if len(shape) != 2:
        #     raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        # if not shape[1]:
        #     raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        # input_size = shape[1]
        # # print(input_size)

        # with tf.variable_scope(scope or "SimpleLinear"):
        #     matrix = tf.get_variable("Matrix4",[ouput_size,input_size],dtype=input_.dtype)
        #     #print(matrix.shape)
        #     bias_term = tf.get_variable("Bias4",[ouput_size],dtype=input_.dtype)
        # return tf.matmul(input_,tf.transpose(matrix))+bias_term 
        shape = input_.get_shape().as_list()
        shape1 = input_y.get_shape().as_list()
        #print(shape)
        #print(ouput_size)
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        input_size1 = shape1[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix4",[ouput_size,input_size],dtype=input_.dtype)
            matrix1 = tf.get_variable("Matrix56565",[ouput_size,input_size1],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias4",[ouput_size],dtype=input_.dtype)
            Wx_plus_b=tf.matmul(input_,tf.transpose(matrix))+tf.matmul(input_y,tf.transpose(matrix1))+bias_term
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b,matrix,matrix1
    def linear5(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix5",[ouput_size,input_size],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias5",[ouput_size],dtype=input_.dtype)
            Wx_plus_b=tf.matmul(input_,tf.transpose(matrix))+bias_term 
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b, matrix
    def linear6(input_,input_y,ouput_size,scope=None):
        # shape = input_.get_shape().as_list()
        # if len(shape) != 2:
        #     raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        # if not shape[1]:
        #     raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        # input_size = shape[1]
        # # print(input_size)

        # with tf.variable_scope(scope or "SimpleLinear"):
        #     matrix = tf.get_variable("Matrix6",[ouput_size,input_size],dtype=input_.dtype)
        #     #print(matrix.shape)
        #     bias_term = tf.get_variable("Bias6",[ouput_size],dtype=input_.dtype)
        # return tf.matmul(input_,tf.transpose(matrix))+bias_term 
        shape = input_.get_shape().as_list()
        shape1 = input_y.get_shape().as_list()
        #print(shape)
        #print(ouput_size)
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        input_size1 = shape1[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix6",[ouput_size,input_size],dtype=input_.dtype)
            matrix1 = tf.get_variable("Matrix73333",[ouput_size,input_size1],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias6",[ouput_size],dtype=input_.dtype)
            Wx_plus_b = tf.matmul(input_,tf.transpose(matrix))+tf.matmul(input_y,tf.transpose(matrix1))+bias_term
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b ,matrix,matrix1


    def linear7(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix7",[ouput_size,input_size],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias7",[ouput_size],dtype=input_.dtype)
            Wx_plus_b=tf.matmul(input_,tf.transpose(matrix))+bias_term
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b ,matrix
    def highway(input_,input_y, size, num_layers=100, bias=-1.0, f=tf.nn.sigmoid, scope='Highway'):
        #print(input_.shape)
        with tf.variable_scope(scope):
            sum0 = 0
            for idx in range(num_layers):
                y,matrix,matrix1 = linear(input_,input_y, size, scope='highway_lin_%d' % idx)
                g = f(y)
                #g1 = f(linear1(input_y, size, scope='highway_lin_%d' % idx))
                #print(g.shape)
                y1,matrix2,matrix3=linear2(input_,input_y, size, scope='highway_gate_%d' % idx)
                sum0 +=tf.norm(matrix,2)+tf.norm(matrix1,2)+tf.norm(matrix2,2)+tf.norm(matrix3,2)
                t = tf.nn.sigmoid(y1)
                #t1 = tf.nn.sigmoid(linear3(input_y, size, scope='highway_gate_%d' % idx) + bias)
                #print(t.shape)
                output = t * g 
                #output1 = t1 * g1 
                #print(output.shape)
                #output = output+ (1. - t) * input_
                output = output+ (1. - t)
                input_ = output
                # output1 = output1+ (1. - t1)
                # input_y = output1
                #print(input_.shape)
            Wx_plus_b = tf.nn.dropout(output, keep_prob)
        return Wx_plus_b,sum0
  
    def decoder(input_,input_y, size,size1, num_layers=100, bias=-1.0, f=tf.nn.sigmoid, scope='Highway'):  
        # y = tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1'])
        # layer_1 = y*tf.nn.sigmoid(y)  
        # y1 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2'])
        # layer_2 = y1*tf.nn.sigmoid(y1) 
        # return layer_2 
        #print(input_.shape)
        with tf.variable_scope(scope):
            sum1=0
            for idx in range(num_layers-1):
                # g = f(linear4(input_, size, scope='highway_lin_%d' % idx))
                # g1 = f(linear5(input_y, size1, scope='highway_lin_%d' % idx))
                # #print(g.shape)
                # t = tf.nn.sigmoid(linear6(input_, size, scope='highway_gate_%d' % idx) + bias)
                # t1 = tf.nn.sigmoid(linear7(input_y, size1, scope='highway_gate_%d' % idx) + bias)
                # #print(t.shape)
                # output = t * g 
                # output1 = t1 * g1 
                # #print(output.shape)
                # #output = output+ (1. - t) * input_
                # output = output+ (1. - t)
                # input_ = output
                # output1 = output1+ (1. - t1)
                # input_y = output1
                #print(input_.shape)
                y,matrix,matrix1=linear4(input_,input_y, 128, scope='highway_lin_%d' % idx)
                g = f(y)
                #g1 = f(linear1(input_y, size, scope='highway_lin_%d' % idx))
                #print(g.shape)
                y1,matrix2,matrix3=linear6(input_,input_y, 128, scope='highway_gate_%d' % idx)
                sum1 +=tf.norm(matrix,2)+tf.norm(matrix1,2)+tf.norm(matrix2,2)+tf.norm(matrix3,2)
                t = tf.nn.sigmoid(y1)
                #t1 = tf.nn.sigmoid(linear3(input_y, size, scope='highway_gate_%d' % idx) + bias)
                #print(t.shape)
                output = t * g 
                #output1 = t1 * g1 
                #print(output.shape)
                #output = output+ (1. - t) * input_
                output = output+ (1. - t)
                input_ = output
                # output1 = output1+ (1. - t1)
                # input_y = output1
                #print(input_.shape)
            score = 100
            y,matrix = linear7(input_,size, scope='highway_lin_%d' % score)
            g1 = f(y)
            # y1,matrix1 = linear1(input_,size, scope='highway_lin_%d' % score)
            # t1 = tf.nn.sigmoid(y1)
            y3,matrix2 = linear5(input_,size1, scope='highway_lin_%d' % score)
            g2 = f(y3)
            # y4,matrix3 = linear3(input_,size1, scope='highway_lin_%d' % score)
            # t2 = tf.nn.sigmoid(y4)
            # output3 = g1*t1+(1. - t1)
            # output1 = g2*t2+(1. -t2)
            output3=g1
            output1=g2
            sum1 +=tf.norm(matrix,2)+tf.norm(matrix2,2)
            #Wx_plus_b = tf.nn.dropout(output3, keep_prob)
            #Wx_plus_b1 = tf.nn.dropout(output1, keep_prob)
            # sum1 +=tf.norm(matrix,2)+tf.norm(matrix1,2)+tf.norm(matrix2,2)+tf.norm(matrix3,2)

        
        return output3,output1,sum1

    # def dropout_selu(x, keep_prob, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
    #              noise_shape=None, seed=None, name=None, training=False):
    #     def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
    #         keep_prob = 1.0 - rate
    #         x = ops.convert_to_tensor(x, name="x")
    #         if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
    #             raise ValueError("keep_prob must be a scalar tensor or a float in the "
    #                                          "range (0, 1], got %g" % keep_prob)
    #         keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
    #         keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    #         alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
    #         keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    #         if tensor_util.constant_value(keep_prob) == 1:
    #             return x

    #         noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
    #         random_tensor = keep_prob
    #         random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
    #         binary_tensor = math_ops.floor(random_tensor)
    #         ret = x * binary_tensor + alpha * (1-binary_tensor)

    #         a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

    #         b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
    #         ret = a * ret + b
    #         ret.set_shape(x.get_shape())
    #         return ret

    #     with ops.name_scope(name, "dropout", [x]) as name:
    #         return utils.smart_cond(training,
    #                             lambda: dropout_selu_impl(x, keep_prob, alpha, noise_shape, seed, name),
    #                             lambda: array_ops.identity(x))
#     def encoder(x,y112):  
#         y = tf.add(tf.add(tf.matmul(x, weights['W1']), tf.matmul(y112, weights['V1'])), biases['encoder_b1'])
#         layer_1 =selu(y)
#         #layer_2 =dropout_selu(layer_1,keep_prob=keep_prob)
#         y1 =tf.add(tf.add(tf.matmul(layer_1, weights['W2']), tf.matmul(y112, weights['V2'])), biases['encoder_b2'])
#         layer_3 =selu(y1)
#         #layer_4 =dropout_selu(layer_3,keep_prob=keep_prob)
#         y2 = tf.add(tf.add(tf.matmul(layer_3, weights['W3']), tf.matmul(y112, weights['V3'])), biases['encoder_b3'])
#         layer_5 =selu(y2)
#         #layer_6 =dropout_selu(layer_5,keep_prob=keep_prob)
#         # y3 =tf.add(tf.add(tf.matmul(layer_3, weights['W4']), tf.matmul(y112, weights['V4'])), biases['encoder_b4'])
#         # layer_4 = tf.nn.sigmoid(y3)

#         return layer_5  
  
  
# # 构建解码器  
#     def decoder(x,y111):  
#         y = tf.add(tf.add(tf.matmul(x, weights['decoder_h12']), tf.matmul(y111, weights['decoder_h42'])), biases['decoder_b111'])
#         layer_1 =selu(y)
#         #layer_2 =dropout_selu(layer_1,keep_prob=keep_prob)  
#         y11 = tf.add(tf.add(tf.matmul(layer_1, weights['decoder_h11']), tf.matmul(y111, weights['decoder_h41'])), biases['decoder_b11'])
#         layer_11 =selu(y11)
#         #layer_4 =dropout_selu(layer_3,keep_prob=keep_prob)
#         # y22 = tf.add(tf.add(tf.matmul(layer_21, weights['decoder_h12']), tf.matmul(y111, weights['decoder_h42'])), biases['decoder_b111'])
#         # layer_31 = tf.nn.sigmoid(y22)
#         y1 = tf.add(tf.matmul(layer_11, weights['decoder_h2']), biases['decoder_b2'])
#         layer_2 = tf.nn.sigmoid(y1) 
#         y2 = tf.add(tf.matmul(layer_11, weights['decoder_h3']), biases['decoder_b3'])
#         layer_3 = tf.nn.sigmoid(y2) 
#         return layer_2,layer_3 
#       
  
# 构建模型  
    # encoder_op,encoder_op1 = highway(X,Y,128) 

    # decoder_op,decoder_side = decoder(encoder_op,encoder_op1,3952,29) 
    encoder_op,sum0 = highway(X_new,Y_new,n_feature) 

    decoder_op,decoder_side,sum1 = decoder(encoder_op,Y,3952,29)  

# 预测  
    y_pred_self = decoder_op
    y_pred_side = decoder_side   
    #print(y_pred)
    y_true = X
    y_true_side = Y
    sum_all=sum0+sum1
# 定义代价函数和优化器 s
    # losse1 = tf.nn.softmax_cross_entropy_with_logits(logits=y_true, labels=y_pred_self)
    # loss1 = tf.reduce_mean(losse1)
    # losse2 = tf.nn.softmax_cross_entropy_with_logits(logits=y_true_side, labels=y_pred_side)
    # loss2 = tf.reduce_mean(losse2)
    cost1 = 0.2*(tf.norm(y_true - y_pred_self, 2)) +0.8*(tf.norm(y_true_side - y_pred_side, 2)) +0.01*(sum_all)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost1) 
    # cost2 = tf.reduce_mean(tf.pow(y_true_side - y_pred_side, 2)) #最小二乘法  
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost2)
    # cost1 = tf.reduce_mean(y_true-tf.log(y_pred_self)) #最小二乘法  
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost1)  
    # cost2 = -tf.reduce_sum(y_true_side*y_pred_side) #最小二乘法  
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost2) 
    #export CUDA_VISIBLE_DEVICES=''
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:  
        
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  
            init = tf.initialize_all_variables()  
        else:  
            init = tf.global_variables_initializer()  
        sess.run(init)  
        
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练  
        total_batch = int(num_examples/batch_size)+1 #总批数 
        for epoch in range(training_epochs):
            j = 0
            for batch in range(total_batch):  
                start_idx = int(batch * batch_size)
                end_idx = int((batch + 1) * batch_size)
                batch_xs = N[start_idx:end_idx,]
                batch_ys = U_side[start_idx:end_idx,]
                batch_xs1 = N[start_idx:end_idx,]
                batch_ys1 = U_side[start_idx:end_idx,]
                for i in range(len(batch_xs1)):
                    a = np.random.uniform(0,1)
                    if a < p:
                        batch_xs1[i] = np.zeros((1,n_input))
                for i in range(len(batch_ys1)):
                    a = np.random.uniform(0,1)
                    if a < p:
                        batch_ys1[i] = np.zeros((1,n_input1))
                _, c1 = sess.run([optimizer,cost1], feed_dict={X: batch_xs,Y: batch_ys, X_new: batch_xs1, Y_new: batch_ys1, keep_prob: 0.8})
                R = sess.run(encoder_op, feed_dict={X_new: batch_xs1, Y_new: batch_ys1, keep_prob: 0.8})

                if j==0:
                    result = R
                else: 
                    result = np.vstack((R,result))
                j = j + 1
            if epoch % display_step == 0:  
                print("loss1 :{}".format(c1))
                #print("loss2 :{}".format(c2))
                # print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)) 
            #print(encoder_op) 
        print("Optimization Finished!")  
  
        return  result




def autoencoder1(N,U_side,n_feature,batch_size,n_input,n_input1,num_examples,training_epochs,learning_rate):

    p = 0.3
    learning_rate = 0.001 
    training_epochs =20
    #batch_size = 151  
    display_step = 1  
    examples_to_show = 10  
    #n_input = 3952  
    #n_input1 = 29 
    #num_examples = 6040
   
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_input1])  
    X_new = tf.placeholder("float", [None, n_input])
    Y_new = tf.placeholder("float", [None, n_input1])    
    keep_prob = tf.placeholder(tf.float32)
  
    # 用字典的方式存储各隐藏层的参数  
    n_hidden_1 = 512 # 第一编码层神经元个数  
    n_hidden_2 = 512 # 第二编码层神经元个数  
    n_hidden_3 = 1024 # 第一编码层神经元个数  
    n_hidden_4 = 200
    # 权重和偏置的变化在编码层和解码层顺序是相逆的  
    # 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数  
    weights = {  
        'W1': tf.Variable(tf.random_uniform(([n_input, n_hidden_1]),minval=0,maxval=1.0,dtype=tf.float32)),  
        'V1': tf.Variable(tf.random_uniform(([n_input1, n_hidden_1]),minval=0,maxval=1.0,dtype=tf.float32)), 
        'W2': tf.Variable(tf.random_uniform(([n_hidden_1, n_hidden_2]),minval=0,maxval=1.0,dtype=tf.float32)),  
        'V2': tf.Variable(tf.random_uniform(([n_input1, n_hidden_2]),minval=0,maxval=1.0,dtype=tf.float32)),
        'W3': tf.Variable(tf.random_uniform(([n_hidden_2, n_hidden_3]),minval=0,maxval=1.0,dtype=tf.float32)),  
        'V3': tf.Variable(tf.random_uniform(([n_input1, n_hidden_3]),minval=0,maxval=1.0,dtype=tf.float32)), 
        'W4': tf.Variable(tf.random_uniform(([n_hidden_3, n_hidden_4]),minval=0,maxval=1.0,dtype=tf.float32)),  
        'V4': tf.Variable(tf.random_uniform(([n_input1, n_hidden_4]),minval=0,maxval=1.0,dtype=tf.float32)),

        'decoder_h1': tf.Variable(tf.random_uniform(([n_hidden_4, n_hidden_3]),minval=0,maxval=1.0,dtype=tf.float32)), 
        'decoder_h11': tf.Variable(tf.random_uniform(([n_hidden_3, n_hidden_2]),minval=0,maxval=1.0,dtype=tf.float32)),
        'decoder_h12': tf.Variable(tf.random_uniform(([n_hidden_2, n_hidden_1]),minval=0,maxval=1.0,dtype=tf.float32)),
        'decoder_h4': tf.Variable(tf.random_uniform(([n_input1, n_hidden_3]),minval=0,maxval=1.0,dtype=tf.float32)), 
        'decoder_h41': tf.Variable(tf.random_uniform(([n_input1, n_hidden_2]),minval=0,maxval=1.0,dtype=tf.float32)), 
        'decoder_h42': tf.Variable(tf.random_uniform(([n_input1, n_hidden_1]),minval=0,maxval=1.0,dtype=tf.float32)),  
        'decoder_h2': tf.Variable(tf.random_uniform(([n_hidden_1, n_input]),minval=0,maxval=1.0,dtype=tf.float32)), 
        'decoder_h3': tf.Variable(tf.random_uniform(([n_hidden_1, n_input1]),minval=0,maxval=1.0,dtype=tf.float32)),  
    }  
    biases = {  
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), 
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),  
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])), 
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),  
        'decoder_b11': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b111': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),  
        'decoder_b3': tf.Variable(tf.random_normal([n_input1])),
    }  

    def selu(x):
        with ops.name_scope('elu') as scope:
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

    def linear(input_,input_y,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        shape1 = input_y.get_shape().as_list()
        #print(shape)
        #print(ouput_size)
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        input_size1 = shape1[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix111",[ouput_size,input_size],dtype=input_.dtype)
            matrix1 = tf.get_variable("Matrix211",[ouput_size,input_size1],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias111",[ouput_size],dtype=input_.dtype)
            Wx_plus_b = tf.matmul(input_,tf.transpose(matrix))+tf.matmul(input_y,tf.transpose(matrix1))+bias_term
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b ,matrix,matrix1
    def linear1(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix31122",[ouput_size,input_size],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias31221",[ouput_size],dtype=input_.dtype)
            Wx_plus_b=tf.matmul(input_,tf.transpose(matrix))+bias_term 
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b, matrix
    def linear2(input_,input_y,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        shape1 = input_y.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        input_size1 = shape1[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix211",[ouput_size,input_size],dtype=input_.dtype)
            matrix1 = tf.get_variable("Matrix311",[ouput_size,input_size1],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias211",[ouput_size],dtype=input_.dtype)
            Wx_plus_b = tf.matmul(input_,tf.transpose(matrix))+tf.matmul(input_y,tf.transpose(matrix1))+bias_term
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b ,matrix,matrix1

    def linear3(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix311",[ouput_size,input_size],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias311",[ouput_size],dtype=input_.dtype)
            Wx_plus_b=tf.matmul(input_,tf.transpose(matrix))+bias_term 
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b, matrix
    def linear4(input_,input_y,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        shape1 = input_y.get_shape().as_list()
        #print(shape)
        #print(ouput_size)
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        input_size1 = shape1[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix411",[ouput_size,input_size],dtype=input_.dtype)
            matrix1 = tf.get_variable("Matrix5111",[ouput_size,input_size1],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias411",[ouput_size],dtype=input_.dtype)
            Wx_plus_b = tf.matmul(input_,tf.transpose(matrix))+tf.matmul(input_y,tf.transpose(matrix1))+bias_term
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b ,matrix,matrix1
    def linear5(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix511",[ouput_size,input_size],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias511",[ouput_size],dtype=input_.dtype)
            Wx_plus_b=tf.matmul(input_,tf.transpose(matrix))+bias_term 
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b, matrix
    def linear6(input_,input_y,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        shape1 = input_y.get_shape().as_list()
        #print(shape)
        #print(ouput_size)
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        input_size1 = shape1[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix6222",[ouput_size,input_size],dtype=input_.dtype)
            matrix1 = tf.get_variable("Matrix72222",[ouput_size,input_size1],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias61111",[ouput_size],dtype=input_.dtype)
            Wx_plus_b = tf.matmul(input_,tf.transpose(matrix))+tf.matmul(input_y,tf.transpose(matrix1))+bias_term
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b ,matrix,matrix1

    def linear7(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix711",[ouput_size,input_size],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias711",[ouput_size],dtype=input_.dtype)
            Wx_plus_b=tf.matmul(input_,tf.transpose(matrix))+bias_term 
            #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        return Wx_plus_b, matrix
    def highway(input_,input_y, size, num_layers=100, bias=-1.0, f=tf.nn.sigmoid, scope='Highway'):
        #print(input_.shape)
        with tf.variable_scope(scope):
            sum0 = 0
            for idx in range(num_layers):
                y,matrix,matrix1 = linear(input_,input_y, size, scope='highway_lin_%d' % idx)
                g = f(y)
                #g1 = f(linear1(input_y, size, scope='highway_lin_%d' % idx))
                #print(g.shape)
                y1,matrix2,matrix3=linear2(input_,input_y, size, scope='highway_gate_%d' % idx)
                sum0 +=tf.norm(matrix,2)+tf.norm(matrix1,2)+tf.norm(matrix2,2)+tf.norm(matrix3,2)
                t = tf.nn.sigmoid(y1)
                output = t * g 
                #output1 = t1 * g1 
                #print(output.shape)
                #output = output+ (1. - t) * input_
                output = output+ (1. - t)
                input_ = output
                # output1 = output1+ (1. - t1)
                # input_y = output1
                #print(input_.shape)
            Wx_plus_b = tf.nn.dropout(output, keep_prob)
        return Wx_plus_b,sum0
  
    def decoder(input_,input_y, size,size1, num_layers=100, bias=-1.0, f=tf.nn.sigmoid, scope='Highway'):  
        # y = tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1'])
        # layer_1 = y*tf.nn.sigmoid(y)  
        # y1 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2'])
        # layer_2 = y1*tf.nn.sigmoid(y1) 
        # return layer_2 
        #print(input_.shape)
        with tf.variable_scope(scope):
            sum1=0
            for idx in range(num_layers-1):
                # g = f(linear4(input_, size, scope='highway_lin_%d' % idx))
                # g1 = f(linear5(input_y, size1, scope='highway_lin_%d' % idx))
                # #print(g.shape)
                # t = tf.nn.sigmoid(linear6(input_, size, scope='highway_gate_%d' % idx) + bias)
                # t1 = tf.nn.sigmoid(linear7(input_y, size1, scope='highway_gate_%d' % idx) + bias)
                # #print(t.shape)
                # output = t * g 
                # output1 = t1 * g1 
                # #print(output.shape)
                # #output = output+ (1. - t) * input_
                # output = output+ (1. - t)
                # input_ = output
                # output1 = output1+ (1. - t1)
                # input_y = output1
                #print(input_.shape)
                y,matrix,matrix1=linear4(input_,input_y, 128, scope='highway_lin_%d' % idx)
                g = f(y)
                #g1 = f(linear1(input_y, size, scope='highway_lin_%d' % idx))
                #print(g.shape)
                y1,matrix2,matrix3=linear6(input_,input_y, 128, scope='highway_gate_%d' % idx)
                sum1 +=tf.norm(matrix,2)+tf.norm(matrix1,2)+tf.norm(matrix2,2)+tf.norm(matrix3,2)
                t = tf.nn.sigmoid(y1)
                #t1 = tf.nn.sigmoid(linear3(input_y, size, scope='highway_gate_%d' % idx) + bias)
                #print(t.shape)
                output = t * g 
                #output1 = t1 * g1 
                #print(output.shape)
                #output = output+ (1. - t) * input_
                output = output+ (1. - t)
                input_ = output
                # output1 = output1+ (1. - t1)
                # input_y = output1
                #print(input_.shape)
            score=1
            y,matrix = linear7(input_,size, scope='highway_lin_%d' % score)
            g1 = f(y)
            # y1,matrix1 = linear1(input_,size, scope='highway_lin_%d' % score)
            # t1 = tf.nn.sigmoid(y1)
            y3,matrix2 = linear5(input_,size1, scope='highway_lin_%d' % score)
            g2 = f(y3)
            # y4,matrix3 = linear3(input_,size1, scope='highway_lin_%d' % score)
            # t2 = tf.nn.sigmoid(y4)
            # output2 = g1*t1+(1. - t1)
            # output1 = g2*t2+(1. -t2)
            output2 = g1
            output1 = g2
            sum1 +=tf.norm(matrix,2)+tf.norm(matrix2,2)
            #Wx_plus_b = tf.nn.dropout(output2, keep_prob)
            #Wx_plus_b1 = tf.nn.dropout(output1, keep_prob)
        return output2,output1,sum1

    # def dropout_selu(x, keep_prob, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
    #              noise_shape=None, seed=None, name=None, training=False):
    #     def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
    #         keep_prob = 1.0 - rate
    #         x = ops.convert_to_tensor(x, name="x")
    #         if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
    #             raise ValueError("keep_prob must be a scalar tensor or a float in the "
    #                                          "range (0, 1], got %g" % keep_prob)
    #         keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
    #         keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    #         alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
    #         keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    #         if tensor_util.constant_value(keep_prob) == 1:
    #             return x

    #         noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
    #         random_tensor = keep_prob
    #         random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
    #         binary_tensor = math_ops.floor(random_tensor)
    #         ret = x * binary_tensor + alpha * (1-binary_tensor)

    #         a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

    #         b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
    #         ret = a * ret + b
    #         ret.set_shape(x.get_shape())
    #         return ret

    #     with ops.name_scope(name, "dropout", [x]) as name:
    #         return utils.smart_cond(training,
    #                             lambda: dropout_selu_impl(x, keep_prob, alpha, noise_shape, seed, name),
    #                             lambda: array_ops.identity(x))
#     def encoder(x,y112):  
#         y = tf.add(tf.add(tf.matmul(x, weights['W1']), tf.matmul(y112, weights['V1'])), biases['encoder_b1'])
#         layer_1 =selu(y)
#         #layer_2 =dropout_selu(layer_1,keep_prob=keep_prob)
#         y1 =tf.add(tf.add(tf.matmul(layer_1, weights['W2']), tf.matmul(y112, weights['V2'])), biases['encoder_b2'])
#         layer_3 =selu(y1)
#         #layer_4 =dropout_selu(layer_3,keep_prob=keep_prob)
#         y2 = tf.add(tf.add(tf.matmul(layer_3, weights['W3']), tf.matmul(y112, weights['V3'])), biases['encoder_b3'])
#         layer_5 =selu(y2)
#         #layer_6 =dropout_selu(layer_5,keep_prob=keep_prob)
#         # y3 =tf.add(tf.add(tf.matmul(layer_3, weights['W4']), tf.matmul(y112, weights['V4'])), biases['encoder_b4'])
#         # layer_4 = tf.nn.sigmoid(y3)

#         return layer_5  
  
  
# # 构建解码器  
#     def decoder(x,y111):  
#         y = tf.add(tf.add(tf.matmul(x, weights['decoder_h12']), tf.matmul(y111, weights['decoder_h42'])), biases['decoder_b111'])
#         layer_1 =selu(y)
#         #layer_2 =dropout_selu(layer_1,keep_prob=keep_prob)  
#         y11 = tf.add(tf.add(tf.matmul(layer_1, weights['decoder_h11']), tf.matmul(y111, weights['decoder_h41'])), biases['decoder_b11'])
#         layer_11 =selu(y11)
#         #layer_4 =dropout_selu(layer_3,keep_prob=keep_prob)
#         # y22 = tf.add(tf.add(tf.matmul(layer_21, weights['decoder_h12']), tf.matmul(y111, weights['decoder_h42'])), biases['decoder_b111'])
#         # layer_31 = tf.nn.sigmoid(y22)
#         y1 = tf.add(tf.matmul(layer_11, weights['decoder_h2']), biases['decoder_b2'])
#         layer_2 = tf.nn.sigmoid(y1) 
#         y2 = tf.add(tf.matmul(layer_11, weights['decoder_h3']), biases['decoder_b3'])
#         layer_3 = tf.nn.sigmoid(y2) 
#         return layer_2,layer_3 
#       
  
# 构建模型  
    encoder_op,sum0 = highway(X_new,Y_new,n_feature) 

    decoder_op,decoder_side,sum1 = decoder(encoder_op,Y,6040,19)  
  
# 预测  
    y_pred_self = decoder_op
    y_pred_side = decoder_side   
    #print(y_pred)
    y_true = X
    y_true_side = Y
    sum_all=sum0+sum1
  
# 定义代价函数和优化器  
    cost1 = 0.2*(tf.norm(y_true - y_pred_self, 2)) +0.8*(tf.norm(y_true_side - y_pred_side, 2))+0.01*sum_all
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost1) 
    # cost2 = tf.reduce_mean(tf.pow(y_true_side - y_pred_side, 2)) #最小二乘法  
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost2)  
    #export CUDA_VISIBLE_DEVICES=''
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:  
        
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  
            init = tf.initialize_all_variables()  
        else:  
            init = tf.global_variables_initializer()  
        sess.run(init)  
        
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练  
        total_batch = int(num_examples/batch_size)+1 #总批数 
        for epoch in range(training_epochs):
            j = 0
            for batch in range(total_batch):  
                start_idx = int(batch * batch_size)
                end_idx = int((batch + 1) * batch_size)
                batch_xs = N[start_idx:end_idx,]
                batch_ys = U_side[start_idx:end_idx,]
                batch_xs1 = N[start_idx:end_idx,]
                batch_ys1 = U_side[start_idx:end_idx,]
                for i in range(len(batch_xs1)):
                    a = np.random.uniform(0,1)
                    if a < p:
                        batch_xs1[i] = np.zeros((1,n_input))
                for i in range(len(batch_ys1)):
                    a = np.random.uniform(0,1)
                    if a < p:
                        batch_ys1[i] = np.zeros((1,n_input1))
                _, c1 = sess.run([optimizer,cost1], feed_dict={X: batch_xs,Y: batch_ys, X_new: batch_xs1, Y_new: batch_ys1, keep_prob: 0.8})
                #_, c2 = sess.run([optimizer2,cost2], feed_dict={Y: batch_ys})
                R = sess.run(encoder_op, feed_dict={X_new: batch_xs1, Y_new: batch_ys1, keep_prob: 0.8})

                if j==0:
                    result = R
                else: 
                    result = np.vstack((R,result))
                j = j + 1
            if epoch % display_step == 0:  
                print("loss1 :{}".format(c1))
                #print("loss2 :{}".format(c2))
                # print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)) 
            #print(encoder_op) 
        print("Optimization Finished!")  
  
        return  result