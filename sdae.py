import tensorflow as tf  
import numpy as np   
import data_helper
# 导入MNIST数据  
# from tensorflow.examples.tutorials.mnist import input_data  
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)  
def autoencoder(N):
    #movie = input_data.read_data_sets("MNIST_data/", one_hot=False)
    learning_rate = 0.08  
    training_epochs = 2
    batch_size = 151  
    display_step = 1  
    examples_to_show = 10  
    n_input = 3952  
    num_examples = 6040
    # tf Graph input (only pictures)  
    X = tf.placeholder("float", [None, n_input])  
  
    # 用字典的方式存储各隐藏层的参数  
    n_hidden_1 = 256 # 第一编码层神经元个数  
    n_hidden_2 = 100 # 第二编码层神经元个数  
    # 权重和偏置的变化在编码层和解码层顺序是相逆的  
    # 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数  
    weights = {  
        'encoder_h1': tf.Variable(tf.random_uniform(([n_input, n_hidden_1]), stddev=0.1)),  
        'encoder_h2': tf.Variable(tf.random_uniform(([n_hidden_1, n_hidden_2]), stddev=0.1)), 
        'decoder_h1': tf.Variable(tf.random_uniform(([n_hidden_2, n_hidden_1]), stddev=0.1)),  
        'decoder_h2': tf.Variable(tf.random_uniform(([n_hidden_1, n_input]), stddev=0.1)),  
    }  
    biases = {  
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),  
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),  
    }  
  
    # 每一层结构都是 xW + b  
    # 构建编码器  
    def encoder(x):  
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),  
                                    biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),  
                                    biases['encoder_b2']))
        #print(layer_2)
        return layer_2  
  
  
# 构建解码器  
    def decoder(x):  
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),  
                                   biases['decoder_b1']))  
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),  
                                   biases['decoder_b2']))  
        return layer_2  
  
    # 构建模型  
    encoder_op = encoder(X) 

    decoder_op = decoder(encoder_op)  
  
    # 预测  
    y_pred = decoder_op  
    #print(y_pred)
    y_true = X  
  
    # 定义代价函数和优化器  
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) #最小二乘法  
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  
  
    with tf.Session() as sess:  
    #print(sess.run(encoder_op),feed_dict={X: batch_xs})
    #tf.initialize_all_variables() no long valid from  
    #2017-03-02 if using tensorflow >= 0.12  
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
                # Run optimization op (backprop) and cost op (to get loss value)  
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                R = sess.run(encoder_op, feed_dict={X: batch_xs})

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
    learning_rate = 0.08  
    training_epochs = 2
    batch_size = 16  
    display_step = 1  
    examples_to_show = 10  
    n_input = 6040  
    num_examples = 3952
    # tf Graph input (only pictures)  
    X = tf.placeholder("float", [None, n_input])  
  
    # 用字典的方式存储各隐藏层的参数  
    n_hidden_1 = 256 # 第一编码层神经元个数  
    n_hidden_2 = 100 # 第二编码层神经元个数  
    # 权重和偏置的变化在编码层和解码层顺序是相逆的  
    # 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数  
    weights = {  
        'encoder_h1': tf.Variable(tf.random_uniform(([n_input, n_hidden_1]), stddev=0.1)),  
        'encoder_h2': tf.Variable(tf.random_uniform(([n_hidden_1, n_hidden_2]), stddev=0.1)), 
        'decoder_h1': tf.Variable(tf.random_uniform(([n_hidden_2, n_hidden_1]), stddev=0.1)),  
        'decoder_h2': tf.Variable(tf.random_uniform(([n_hidden_1, n_input]), stddev=0.1)),  
    }  
    biases = {  
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),  
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),  
    }   
  
    # 每一层结构都是 xW + b  
    # 构建编码器  
    def encoder(x):  
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),  
                                    biases['encoder_b1']))  
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),  
                                    biases['encoder_b2']))  
        #print(layer_2)
        return layer_2  
  
  
# 构建解码器  
    def decoder(x):  
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),  
                                   biases['decoder_b1']))  
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),  
                                   biases['decoder_b2']))  
        return layer_2  
  
# 构建模型  
    encoder_op = encoder(X) 

    decoder_op = decoder(encoder_op)  
  
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
                batch_xs = NT[start_idx:end_idx,] 
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