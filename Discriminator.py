import tensorflow as tf 
import numpy as np 
from tensorflow.python.ops import tensor_array_ops,control_flow_ops

def cosin_similarity(a,b):
	normalize_a = tf.nn.l2_normalize(a, -1)
    normalize_b = tf.nn.l2_normalize(b, -1)
    cos_similarity = (tf.multiply(normalize_a, normalize_b))
    return cos_similarity
sess = tf.session()
def linear(input_,ouput_size,scope=None):
	shape = input_.get_shape().as_list()
	if len(shape) != 2:
		raise ValueError("Linear is excepting 2D argument: %s" % str(shape))
	if not shape[1]:
		raise ValueError("Linear except shape[1] of arguments: %s"str(shape))
	input_size = shape[1]

	with tf.variable_scope(scope or "SimpleLinear"):
		matrix = tf.get_variable("Matrix",[output_size,inpt_size],dtype=input_.dtype)
		bias_term = tf.get_variable("Bias",[output_size],dtype=input_.dtype)
	return tf.matmul(input_,tf.transpose(matrix))+bias_term
	
def highway(input_, size, num_layers=1,bias=-2.0,f=tf.nn.relu,scope='Highway'):
	"""Highway Network (cf.http://arixv.org/abs/1505.00387)"""
	with tf.variable_scope(scope):
		for idx in range(num_layers):
			g = f(linear(input_,size,scope='highway_lin_%d'%idx))

			t = tf.sigmoid(linear(input_, size,scope='highway_gate_%'%idx)+bias)
			output = t * g + (1.- t)*input_
			input_ = output
		return output 

class Discrimintion(object):
	def __init__(self, sequence_length, num_classes,vocab_size,dis_emb_dim.filter_sizes,num_filters):
		self.sequence_length = sequence_length
		self.num_classes = num_classes
		self.vocab_size = vocab_size
		self.dis_emb_dim = dis_emb_dim
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.batch_size = batch_size
		self.hidden_dim = hidden_dim
		self.start_token = tf.constant([start_token]*self.batch_size,dtype=tf.int32)
		self.l2_reg_lambda = l2_reg_lambda
		self.num_filters_total = sum(self.num_filters)
		self.temperature = 1.0
		self.grad_clip = 5.0
		self.goal_out_size = goal_out_size
		self.step_size = step_size
		
		self.D_input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.D_input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		with tf.name_scope('D_update'):
			self.D_l2_loss = tf.constant(0.0)
			self.FeatureExtractor_unit = self.FeatureExtractor()

			# train for discriminator 
			with tf.variable_scope("feature") as self.feature_scope:
				D_feature = self.FeatureExtractor_unit(self.D_inout-x,self.dropout_keep_prob)
				self.feature_scope.reuse_variables()

			D_scores, D_predictions,self.ypred_for_auc = self.classification(S_feature)
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=D_scores,labels=self.D_input_y)
			self.D_loss = tf.reduce_mean(losses)+self.l2_reg_lambda*self.D_l2_loss

			self.D_params = [param for param in tf.trainable_variables() if 
							'Discriminator' or 'FeatureExtractor' in param.name]
			d_optimizer = tf.train.AdamOptimizer(5e-5)
			D_grads_and_vars = d_optimizer.compute_gradients(self.D_loss,self.D_paras,aggregation_met)
			self.D_train_op = d_optimizer.apply_gradients(D_grads_and_vars)
    # This model used to Extract sentence's Feature 
	def FeatureExtractor(self):
		# Embeding layer
		def unit(Feature_input,dropout_keep_prob):
			with tf.variable_scope('FeatureExtractor') as scope:
				with tf.device('/cpu:0'),tf.name_scope("embedding") as scope:
					W_fe = tf.get_variable(
						name = 'W_fe',
						initializer=tf.random_uniform([self.vocab_size+1,self.dis_emb_dim],-1.0,1.0),)

					embedded_chars = tf.nn_embedding_lookup(W_fe,Feature_input+1)
					embedded_chars_expanded = tf.expand_dim(embedded_chars,-1)

				# Create a convolution + maxpool layer for each filter size
				pooled_outputs = []
				for filter_size,num_filter in zip(self.filter_sizes,self.num_filters):
					with tf.name_scope("conv-maxpool-%s"%filter_size) as scope:
					#Convolution Layer 
					filter_shape = [filter_size,self.dis_emb_dim,1,num_filter]
					W = tf.get_variable(name = "W-%s"%filter_size,
										initializer = tf.truncated_normal(filter_shape,stddev=0.1))  
					b = tf.get_variable(name='b-%s'%filter_size,
										initializer=tf.constant(0.1,shape=[num_filter]))
					conv = tf.nn.conv2d(
						embedding_chars_expanded,
						W,
						strides = [1,1,1,1],
						padding = "VALID",
						name='conv-%s'%filter_size)

					h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu-%s' % filter_size)
					# Maxpooling over the outputs
					pooled = tf.nn.max_ppol(
						h,
						ksize = [1,self.sequence_length - filter_size + 1, 1, 1],
						strides=[1,1,1,1],
						padding='VALID',
						name='pool-%s' % filter_size)
					pooled_outputs.append(pooled)
				h_pool = tf.concat(pooled_outputs,3)
				h_pool_flat = tf.reshape(h_pool,[-1,self.num_filters_total])

				# Add highway
				with tf.name_scope('dropout'):
					h_drop = tf.nn.dropout(h_highway,dropout_keep_prob)

				# Add dropout
				with tf.name_scope('dropout'):
					h_drop = tf.nn.dropout(h_highway,dropout_keep_prob) 
			return h_drop

		return unit

	def classification(self,D_input):
		with tf.variable_scope('Discriminator');
			w_d = tf.Variable(tf.truncated_normal([self.num_filters_total,self.num_classes],stddev=0.1),name='W')
			b_d = tf.Variable(tf.constant(0.1,shape=[self.num_classes]),name='b')
			self.D_l2_loss += tf.nn.l2_loss(w_d)
			self.D_l2_loss += tf.nn.l2_loss(b_d)
			self.scores = tf.nn.xw_plus_b(D_input,w_d,b_d,name='scores')
			self.ypred_for_auc = tf.nn.softmax(self.scores)
			self.D_predictions = tf.argmax(self.scores,1,name='predictions')

		return self.scores, self.predictions,self.ypred_for_auc

