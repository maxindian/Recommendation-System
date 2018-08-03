import tensorflow as tf  
import numpy as np   
import data_helper
# 导入MNIST数据  
# from tensorflow.examples.tutorials.mnist import input_data  
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)  
def autoencoder(N):
    #movie = input_data.read_data_sets("MNIST_data/", one_hot=False)
    p = 0.5
    learning_rate = 0.01  
    training_epochs =20
    batch_size = 151  
    display_step = 1  
    examples_to_show = 10  
    n_input = 29 
    num_examples = 6040
    # tf Graph input (only pictures)  
    X = tf.placeholder("float", [None, n_input])  
  
    # 用字典的方式存储各隐藏层的参数  
    n_hidden_1 = 256 # 第一编码层神经元个数  
    n_hidden_2 = 100 # 第二编码层神经元个数  
    # 权重和偏置的变化在编码层和解码层顺序是相逆的  
    # 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数  
    weights = {  
        'encoder_h1': tf.Variable(tf.random_uniform(([n_input, n_hidden_1]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
        'encoder_h2': tf.Variable(tf.random_uniform(([n_hidden_1, n_hidden_2]),minval=-1.0,maxval=1.0,dtype=tf.float32)), 
        'decoder_h1': tf.Variable(tf.random_uniform(([n_hidden_2, n_hidden_1]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
        'decoder_h2': tf.Variable(tf.random_uniform(([n_hidden_1, n_input]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
    }  
    biases = {  
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),  
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),  
    }  
  
    # 每一层结构都是 xW + b  
    # 构建编码器  
#     def encoder(x):  
#         y = tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1'])
#         layer_1 = y*tf.nn.sigmoid(y) 
#         y1 =tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2'])
#         layer_2 = y1*tf.nn.sigmoid(y1)
#         #print(layer_2)
#         return layer_2  
  
    def linear(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        #print(shape)
        #print(ouput_size)
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]
        # print(input_size)

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix4",[ouput_size,input_size],dtype=input_.dtype)
            #print(matrix.shape)
            bias_term = tf.get_variable("Bias4",[ouput_size],dtype=input_.dtype)
        return tf.matmul(input_,tf.transpose(matrix))+bias_term  
    def linear1(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        #print(shape)
        #print(ouput_size)
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
        return tf.matmul(input_,tf.transpose(matrix))+bias_term  
# # 构建解码器  
#     def decoder(x):
#         y =tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1'])
#         layer_1 = y*tf.nn.sigmoid(y)
#         y1 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2'])
#         layer_2 = y1*tf.nn.sigmoid(y1)
#         return layer_2  
    def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
        #print(input_.shape)
        with tf.variable_scope(scope):
            for idx in range(num_layers):
                g = f(linear(input_, size, scope='highway_lin_%d' % idx))
                #print(g.shape)
                t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
                #print(t.shape)
                output = t * g 
                #print(output.shape)
                #output = output+ (1. - t) * input_
                output = output+ (1. - t)
                input_ = output
                #print(input_.shape)
        return output  
  
  
# 构建解码器  
    def decoder(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):  
        # y = tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1'])
        # layer_1 = y*tf.nn.sigmoid(y)  
        # y1 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2'])
        # layer_2 = y1*tf.nn.sigmoid(y1) 
        # return layer_2 
        #print(input_.shape)
        with tf.variable_scope(scope):
            for idx in range(num_layers):
                g = f(linear1(input_, size, scope='highway_lin_%d' % idx))
                print(g.shape)
                t = tf.sigmoid(linear1(input_, size, scope='highway_gate_%d' % idx) + bias)
                print(t.shape)
                output = t * g 
                print(output.shape)
                #output = output+ (1. - t) * input_
                output = output+ (1. - t)
                input_ = output
                print(input_.shape)
        return output
  
# 构建模型  
    encoder_op = highway(X,100)
    decoder_op = decoder(encoder_op,29)  
  
# 预测  
    y_pred = decoder_op  
    #print(y_pred)
    y_true = X  
  
# 定义代价函数和优化器  
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) #最小二乘法  
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  
  
    with tf.Session() as sess:  
        #print(sess.run(encoder_op),feed_dict={X: batch_xs})
    # tf.initialize_all_variables() no long valid from  
    # 2017-03-02 if using tensorflow >= 0.12  
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  
            init = tf.initialize_all_variables()  
        else:  
            init = tf.global_variables_initializer()  
        sess.run(init)  
        
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练  
        total_batch = int(num_examples/batch_size) #总批数  
        for epoch in range(training_epochs):
            j = 0
            for batch in range(total_batch):  
                start_idx = int(batch * batch_size)
                end_idx = int((batch + 1) * batch_size)
                batch_xs = N[start_idx:end_idx,]
                #print(batch_xs)
                #print(type(batch_xs[0]))
                # for i in range(len(batch_xs)):
                #     a = np.random.uniform(0,1)
                #     if a < p:
                #         batch_xs[i] = np.zeros((1,3952))
                #print(batch_xs)
                # Run optimization op (backprop) and cost op (to get loss value)  
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                R = sess.run(encoder_op, feed_dict={X: batch_xs})
                #print(R)
                if j==0:
                    result = R
                else: 
                    result = np.vstack((R,result))
                j = j + 1
                #print(R.shape)
            # batch_xs=[]
            # for i in range(count,data_size):
            #     batch_xs.append(movie[i])
            # R, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            # R = sess.run(encoder_op, feed_dict={X: batch_xs})
            # if j==0:
            #     result = R
            # else: 
            #     result = np.vstack((result,R))
            # j = j + 1
            # #tf.print(encoder_op,)
            if epoch % display_step == 0:  
                print("loss :{}".format(c))
                # print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)) 
            #print(encoder_op) 
        print("Optimization Finished!")  
  
        return  result

def autoencoder1(NT):
    #movie = input_data.read_data_sets("MNIST_data/", one_hot=False)
    learning_rate = 0.01  
    training_epochs = 20
    batch_size = 16  
    display_step = 1  
    examples_to_show = 10  
    n_input = 19  
    num_examples = 3952
    p = 0.5
    # tf Graph input (only pictures)  
    X = tf.placeholder("float", [None, n_input])  
  
    # 用字典的方式存储各隐藏层的参数  
    n_hidden_1 = 256 # 第一编码层神经元个数  
    n_hidden_2 = 100 # 第二编码层神经元个数  
    # 权重和偏置的变化在编码层和解码层顺序是相逆的  
    # 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数  
    weights = {  
        'encoder_h1': tf.Variable(tf.random_uniform(([n_input, n_hidden_1]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
        'encoder_h2': tf.Variable(tf.random_uniform(([n_hidden_1, n_hidden_2]),minval=-1.0,maxval=1.0,dtype=tf.float32)), 
        'decoder_h1': tf.Variable(tf.random_uniform(([n_hidden_2, n_hidden_1]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
        'decoder_h2': tf.Variable(tf.random_uniform(([n_hidden_1, n_input]),minval=-1.0,maxval=1.0,dtype=tf.float32)),  
    }  
    biases = {  
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),  
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),  
    }   
  
    # 每一层结构都是 xW + b  
    # 构建编码器  
#     def encoder(x):  
#         y = tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1'])
#         layer_1 = y*tf.nn.sigmoid(y)
#         y1 =tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2'])
#         layer_2 = y1*tf.nn.sigmoid(y1)
#         return layer_2  
  
  
# # 构建解码器  
#     def decoder(x):  
#         y =tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1'])
#         layer_1 = y*tf.nn.sigmoid(y)  
#         y1 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2'])
#         layer_2 = y1*tf.nn.sigmoid(y1) 
#         return layer_2  
    def linear(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        #print(shape)
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix7",[ouput_size,input_size],dtype=input_.dtype)
            bias_term = tf.get_variable("Bias7",[ouput_size],dtype=input_.dtype)
        return tf.matmul(input_,tf.transpose(matrix))+bias_term  
    def linear1(input_,ouput_size,scope=None):
        shape = input_.get_shape().as_list()
        #print(shape)
        if len(shape) != 2:
            raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear except shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]

        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix6",[ouput_size,input_size],dtype=input_.dtype)
            bias_term = tf.get_variable("Bias6",[ouput_size],dtype=input_.dtype)
        return tf.matmul(input_,tf.transpose(matrix))+bias_term  
# # 构建解码器  
#     def decoder(x):
#         y =tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1'])
#         layer_1 = y*tf.nn.sigmoid(y)
#         y1 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2'])
#         layer_2 = y1*tf.nn.sigmoid(y1)
#         return layer_2  
    def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
        #print(input_.shape)
        with tf.variable_scope(scope):
            for idx in range(num_layers):
                g = f(linear(input_, size, scope='highway_lin_%d' % idx))
                #print(g.shape)
                t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
                #print(t.shape)
                output = t * g 
                #print(output.shape)
                #output = output+ (1. - t) * input_
                output = output+ (1. - t)
                input_ = output
                #print(input_.shape)
        return output  
  
  
# 构建解码器  
    def decoder(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):  
        # y = tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1'])
        # layer_1 = y*tf.nn.sigmoid(y)  
        # y1 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2'])
        # layer_2 = y1*tf.nn.sigmoid(y1) 
        # return layer_2 
        #print(input_.shape)
        with tf.variable_scope(scope):
            for idx in range(num_layers):
                g = f(linear1(input_, size, scope='highway_lin_%d' % idx))
                #print(g.shape)
                t = tf.sigmoid(linear1(input_, size, scope='highway_gate_%d' % idx) + bias)
                #print(t.shape)
                output = t * g 
                #print(output.shape)
                #output = output+ (1. - t) * input_
                output = output+ (1. - t)
                input_ = output
                #print(input_.shape)
        return output
  
# 构建模型  
    encoder_op = highway(X,100) 
    decoder_op = decoder(encoder_op,19)  
  
# 预测  
    y_pred = decoder_op  
    #print(y_pred)
    y_true = X  
  
# 定义代价函数和优化器  
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) #最小二乘法  
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    with tf.Session(config=config) as sess:  
        #print(sess.run(encoder_op),feed_dict={X: batch_xs})
    # tf.initialize_all_variables() no long valid from  
    # 2017-03-02 if using tensorflow >= 0.12  
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  
            init = tf.initialize_all_variables()  
        else:  
            init = tf.global_variables_initializer()  
        sess.run(init)  
        
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练  
        total_batch = int(num_examples/batch_size) #总批数 
        for epoch in range(training_epochs):
            j = 0
            for batch in range(total_batch):
                start_idx = int(batch * batch_size)
                end_idx = int((batch + 1) * batch_size)
                batch_xs = NT[start_idx:end_idx,]
                # for i in range(len(batch_xs)):
                #     a = np.random.uniform(0,1)
                #     if a < p:
                #         batch_xs[i] = np.zeros((1,6040))
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs}) 
                
                R_item = sess.run(encoder_op, feed_dict={X: batch_xs})
                if j==0:
                    result = R_item
                else: 
                    result = np.vstack((R_item,result))
                j = j + 1
            # batch_xs=[]
            # for i in range(count,data_size):
            #     batch_xs.append(user[i])
            # _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            # R_user = sess.run(encoder_op, feed_dict={X: batch_xs})
            
            # if j==0:
            #     result = R_user
            # else: 
            #     result = np.vstack((result,R_user))
            # j = j + 1
            #tf.print(encoder_op,)
            if epoch % display_step == 0:  
                print("loss :{}".format(c))
                #print("op :{}".format(o))
                # print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)) 
            #print(encoder_op) 
        print("Optimization Finished!")  
  
        return  result