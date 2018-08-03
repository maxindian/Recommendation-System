# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from six.moves import urllib
from tensorflow.contrib import learn
from numpy.random import RandomState
from scipy import sparse
from scipy.sparse import lil_matrix
import autoencoder_highway
import autoencoder_highway1
import autoencoder
import autoencoder1
import autoencoder_aaai12
#from PMF import PMF
from evaluation import RMSE
#from PMF_12 import PMF
from BPMF import BPMF 
from Optimizer1 import *
#from Optimizer1 import * 
from numpy import linalg as LA
from PCA import *
import side_information 
import pandas as pd
import pickle
def com_f1(N):
	f1_dic = {}
	#user_num = []
	for i in range(N.shape[0]):
		if sum(N[i]) <=10:
			f1_dic[i]=[0]
		else:
			f1_dic[i]=[sum(N[i])]
		#user_num.append(sum(N[i]))
		for j in range(N.shape[1]):
			if N[i][j] == 1:
				f1_dic[i].append(j)
	return f1_dic

def com_f2(N1,Top_K):
	f2_dic = {}
	for i in range(N1.shape[0]):
		f2_dic[i] = []
		sub_dic = {}
		
		for key,eva in enumerate(N1[i]):
			sub_dic[key] = eva
		sub_tuple=sorted(sub_dic.items(),key=lambda item:item[1],reverse=True)
		
		for j in range(Top_K):
			f2_dic[i].append(sub_tuple[j][0])
	return f2_dic
def compute_recall(f1_dic,f2_dic):
	#count_sum0 = 0
	toatal_recall = 0 # the sum of all user's recall values 
	for i in range((len(f1_dic))):
		sub_recall = 0 # every user's recall value
		count_n = 0
		taltal_num = int(f1_dic[i][0])
		if taltal_num != 0:
			# len(data2[i])
			for j in range(len(f2_dic[i])):  # 这里的值可以是taltal_num
				if f2_dic[i][j] in f1_dic[i][1:]:
					count_n += 1
			sub_recall = count_n/taltal_num
			
		else:
			sub_recall = 1
			#count_sum0 += 1
		toatal_recall += sub_recall
	recall = toatal_recall/(len(f1_dic))
	return recall
List_top_K=[50,100,150,200,250,300]
# rating_data_file='ml-1m/ml-1m/ratings.dat'
# R = data_helper.construct_R(rating_data_file)
# for i in range(6040):
# 	for j in range(3952):
# 		print(R[i][j])
iter1 = 20
rand_state = RandomState(0)
num_user = 6040
num_item = 3952
rating_data_file='ml-1m/ml-1m/ratings.dat'
M = lil_matrix((num_user, num_item),dtype=np.float32)#?1
total_number= {}
file_object = open('Orignial_R.txt','wb')
with open(rating_data_file, 'r') as f:
	for line in f.readlines():
		tokens = line.split("::")
		user_id = int(tokens[0]) - 1  # 0 base index
		item_id = int(tokens[1]) - 1
		rating = int(tokens[2])
		#rating = (rating-1)/4
		#M[user_id, item_id] = 1
		if rating>=5:
			M[user_id, item_id] = 1
		else:
			M[user_id, item_id] = 0
N = M.toarray()
# for i in range(N.shape[0]):
# 	total_number[i] = [sum(M[i])]
# 	for j in range(N.shape[1]):
# 		if N[i][j] == 1:
# 			total_number[i].append(j)
# #file_object.write(total_number)
# pickle.dump(total_number,file_object)
# exit()
N = M.toarray()
NT = N.T
f1_dic=com_f1(N)
ratings = data_helper.load_movielens_ratings(rating_data_file)
#ratings = int(ratings[:,(0,1)])
n_user = max(ratings[:, 0])
#n_user = int(n_user)
n_item = max(ratings[:, 1])

#n_item = int(n_item)
# shift user_id & movie_id by 1. let user_id & movie_id start from 0
ratings[:, (0, 1)] -= 1
#print(ratings)
#print(M)
# split data to training & testing
train_pct = 0.8
#rand_state.shuffle(ratings)
train_size = int(train_pct * ratings.shape[0])

train = ratings[:train_size]
validation = ratings[train_size:]
n_feature = 128
#N = N[:4832,:]
# R = load_file() 
# R_ = Optimizer(R)
# R_ = np.array(R_)

# R_T = R_.T
# print(R_[1])
# #for iteration in iter1:
# U = autoencoder1.autoencoder(N)
# V = autoencoder1.autoencoder1(NT)

#辅助信息
U_side = side_information.user()
V_side = side_information.item()

# print(N)
# print(NT)
#方法1：将本身信息和辅助信息分别利用autoencoder
# U = autoencoder_highway.autoencoder(N)
# V = autoencoder_highway.autoencoder1(NT)
# U_side_all = autoencoder_highway1.autoencoder(U_side)
# V_side_all = autoencoder_highway1.autoencoder1(V_side)
# U = autoencoder_aaai.autoencoder(N,U_side)
# V = autoencoder_aaai.autoencoder1(NT,V_side)




# V = autoencoder_aaai.autoencoder1(NT,V_side,n_feature=128,batch_size=128,n_input=6040,n_input1=19,num_examples=3952,training_epochs=20,learning_rate=0.001)
# U = autoencoder_aaai.autoencoder(N,U_side,n_feature=128,batch_size=128,n_input=3952,n_input1=29,num_examples=6040,training_epochs=20,learning_rate=0.001)

# # U_alVl = np.zeros((6040,100))
# # V_all = np.zeros((3952,100))
# # for i in range(U_side_all.shape[0]):
# # 	for j in range(U_side_all.shape[1]):
# # 		if U[i][j]>U_side_all[i][j]:
# # 			U_all[i][j] = U[i][j]
# # 		else:
# # 			U_all[i][j] = U_side_all[i][j]

# # for i in range(V_side_all.shape[0]):
# # 	for j in range(V_side_all.shape[1]):
# # 		if V[i][j]>V_side_all[i][j]:
# # 			V_all[i][j] = V[i][j]
# # 		else:
# # 			V_all[i][j] = V_side_all[i][j]
# # for  i in range(U.shape[0]):
# # 	for j in range(U.shape[1]):
# # 		if U[i][j] < 0.0001:
# # 			U[i][j] = np.random.uniform(0,0.03)
# # 		if U[i][j] > 0.1:
# # 			U[i][j] = np.random.uniform(0,0.3)
# # for  i in range(V.shape[0]):
# # 	for j in range(V.shape[1]):
# # 		if V[i][j] < 0.0001:
# # 			V[i][j] = np.random.uniform(0,0.03)
# # 		if V[i][j] >0.1:
# # 			V[i][j] = np.random.uniform(0,0.3)
# eval_iters = 10
# # bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_feature,U=U,V=V,
# #             max_rating=5.0, min_rating=0.0, seed=0).fit(train, n_iters=eval_iters)
# #bpmf.Q_learning()

# R = np.matmul(U, V.T)
# #R_m = np.matrix(R_)
# print("==========================================")
# #分别对U和V进行强化学习
# R = pd.DataFrame(R)
# print("load data")

# R_ = Optimizer(R)
# R_ = np.array(R_)
# R_m = np.matrix(R_)
# print("==========================================")
# # # U = np.matrix(U)
# # # U = U.I
# # # V = np.dot(U,R_m)
# # # V1 = V.I
# # # U = np.dot(R_m,V1)

# # # U = np.array(U)
# # # V = np.array(V)
# # # V = V.T
# # # print(V.shape)
# # # print(U.shape)
# U = pca(R_m,128)
# V = pca(R_m.T,128)
# U = np.array(U)
# V = np.array(V)
# print(U.shape)
# print(V.shape)
# # steps = 100
# # alpha = 0.0002
# # beta = float(0.02)
# # #U,V = matrix_factorization(R_m,U,V,steps,alpha,beta)

# # def matrix_factorization(X,P,Q,K,steps,alpha,beta):
# #     Q = Q.T
# #     for step in range(steps):
# #         print (step)
# #         #for each user
# #         for i in range(X.shape[0]):
# #             #for each item
# #             for j in range(X.shape[1]):
# #             	#print(i)
# #             	if X[i][j] > 0: #calculate the error of the element
# #                     eij = X[i][j] - np.dot(P[i,:],Q[:,j])
# #                     #second norm of P and Q for regularilization
# #                     sum_of_norms = 0
# #                     #for k in xrange(K):
# #                     #    sum_of_norms += LA.norm(P[:,k]) + LA.norm(Q[k,:])
# #                     #added regularized term to the error
# #                     sum_of_norms += LA.norm(P) + LA.norm(Q)
# #                     #print sum_of_norms
# #                     eij += ((beta/2) * sum_of_norms)
# #                     #print eij
                    
# #                     #compute the gradient from the error
# #                     for k in range(K):
# #                         P[i][k] = P[i][k] + alpha * ( 2 * eij * Q[k][j] - (beta * P[i][k]))
# #                         Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - (beta * Q[k][j]))

# #         #compute total error
# #         error = 0
# #         #for each user
# #         for i in range(X.shape[0]):
# #             #for each item
# #             for j in range(X.shape[1]):
# #                 if X[i][j] > 0:
# #                     error += np.power(X[i][j] - np.dot(P[i,:],Q[:,j]),2)
# #         if error < 0.001:
# #             break
# #     return P, Q.T
# #U,V = matrix_factorization(R_,U,V,n_feature,steps,alpha,beta)
# #eval_iters = 50
# bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_feature,U=U,V=V,
#             max_rating=1.0, min_rating=0.0, seed=0).fit(train, n_iters=eval_iters)
# #mean_rating_ = np.mean(validation[:, 2])
# # def predict(user_features_,item_features_, data,max_rating=5.0,min_rating=1.0):
# # 	if not mean_rating_:
# # 		raise NotFittedError()
# # 	u_features = user_features_.take(data.take(0, axis=1), axis=0)
# # 	i_features = item_features_.take(data.take(1, axis=1), axis=0)
# # 	preds = np.sum(u_features * i_features, 1) + mean_rating_
# # 	if max_rating:
# # 		preds[preds > max_rating] = max_rating
# # 	if min_rating:
# # 		preds[preds < min_rating] = min_rating
# # 	return preds
# # train_preds = bpmf.predict(ratings[:, :2])
# # train_rmse = RMSE(train_preds, ratings[:, 2])
# #bpmf.fit(validation, n_iters=eval_iters)
# val_preds = bpmf.predict(validation[:, :2])
# val_rmse = RMSE(val_preds, validation[:, 2])
# print("after %d iteration, validation RMSE: %.6f" %
#       (eval_iters,val_rmse))
V = autoencoder_aaai12.autoencoder1(NT,V_side,n_feature=128,batch_size=128,n_input=6040,n_input1=19,num_examples=3952,training_epochs=20,learning_rate=0.001)
U = autoencoder_aaai12.autoencoder(N,U_side,n_feature=128,batch_size=128,n_input=3952,n_input1=29,num_examples=6040,training_epochs=20,learning_rate=0.001)
for i in range(10):
	print(i)
	val_rmse=0.0
	if i<=0:
		# U_alVl = np.zeros((6040,100))
		# V_all = np.zeros((3952,100))
		# for i in range(U_side_all.shape[0]):
		# 	for j in range(U_side_all.shape[1]):
		# 		if U[i][j]>U_side_all[i][j]:
		# 			U_all[i][j] = U[i][j]
		# 		else:
		# 			U_all[i][j] = U_side_all[i][j]

		# for i in range(V_side_all.shape[0]):
		# 	for j in range(V_side_all.shape[1]):
		# 		if V[i][j]>V_side_all[i][j]:
		# 			V_all[i][j] = V[i][j]
		# 		else:
		# 			V_all[i][j] = V_side_all[i][j]
		# for  i in range(U.shape[0]):
		# 	for j in range(U.shape[1]):
		# 		if U[i][j] < 0.0001:
		# 			U[i][j] = np.random.uniform(0,0.03)
		# 		if U[i][j] > 0.1:
		# 			U[i][j] = np.random.uniform(0,0.3)
		# for  i in range(V.shape[0]):
		# 	for j in range(V.shape[1]):
		# 		if V[i][j] < 0.0001:
		# 			V[i][j] = np.random.uniform(0,0.03)
		# 		if V[i][j] >0.1:
		# 			V[i][j] = np.random.uniform(0,0.3)
		eval_iters = 100
		# bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_feature,U=U,V=V,
		#             max_rating=5.0, min_rating=0.0, seed=0).fit(train, n_iters=eval_iters)
		#bpmf.Q_learning()

		R = np.matmul(U, V.T)
		#R_m = np.matrix(R_)
		print("==========================================")
		#分别对U和V进行强化学习
		R = pd.DataFrame(R)
		print("load data")

		R_ = Optimizer1(R)
		R_ = np.array(R_)
		R_m = np.matrix(R_)
		print("==========================================")
		# # # U = np.matrix(U)
		# # # U = U.I
		# # # V = np.dot(U,R_m)
		# # # V1 = V.I
		# # # U = np.dot(R_m,V1)

		# # # U = np.array(U)
		# # # V = np.array(V)
		# # # V = V.T
		# # # print(V.shape)
		# # # print(U.shape)
		U = pca(R_m,128)
		V = pca(R_m.T,128)
		U = np.array(U)
		V = np.array(V)
		print(U.shape)
		print(V.shape)
		# steps = 100
		# alpha = 0.0002
		# beta = float(0.02)
		# #U,V = matrix_factorization(R_m,U,V,steps,alpha,beta)

		# def matrix_factorization(X,P,Q,K,steps,alpha,beta):
		#     Q = Q.T
		#     for step in range(steps):
		#         print (step)
		#         #for each user
		#         for i in range(X.shape[0]):
		#             #for each item
		#             for j in range(X.shape[1]):
		#             	#print(i)
		#             	if X[i][j] > 0: #calculate the error of the element
		#                     eij = X[i][j] - np.dot(P[i,:],Q[:,j])
		#                     #second norm of P and Q for regularilization
		#                     sum_of_norms = 0
		#                     #for k in xrange(K):
		#                     #    sum_of_norms += LA.norm(P[:,k]) + LA.norm(Q[k,:])
		#                     #added regularized term to the error
		#                     sum_of_norms += LA.norm(P) + LA.norm(Q)
		#                     #print sum_of_norms
		#                     eij += ((beta/2) * sum_of_norms)
		#                     #print eij
		                    
		#                     #compute the gradient from the error
		#                     for k in range(K):
		#                         P[i][k] = P[i][k] + alpha * ( 2 * eij * Q[k][j] - (beta * P[i][k]))
		#                         Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - (beta * Q[k][j]))

		#         #compute total error
		#         error = 0
		#         #for each user
		#         for i in range(X.shape[0]):
		#             #for each item
		#             for j in range(X.shape[1]):
		#                 if X[i][j] > 0:
		#                     error += np.power(X[i][j] - np.dot(P[i,:],Q[:,j]),2)
		#         if error < 0.001:
		#             break
		#     return P, Q.T
		#U,V = matrix_factorization(R_,U,V,n_feature,steps,alpha,beta)
		#eval_iters = 50
		bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_feature,U=U,V=V,
		            max_rating=1.0, min_rating=0.0, seed=0).fit(ratings, n_iters=eval_iters)
		N1=bpmf.R_predict()
		for k in List_top_K:
			f2_dic=com_f2(N1,k)
			recall=compute_recall(f1_dic,f2_dic)
			print("Top_K : %d " % k)
			print("top_k recall's value is : %.4f" % recall)
		# R_new=bpmf.R_predict()
		# K = 5 # top K
		# file1_object = open('New_R.txt','wb')
		# total_number={}
		# sub = {}
		# for i in range(R_new.shape[0]):
		# 	for j in range(R_new.shape[1]):
		# 		sub[j] =  R_new[i][j] 
		# 	sub_after = dict(sorted(sub.items(),key=lambda v:v[1]))
		# 	count = 0
		# 	total_number[i]=[]
		# 	for key,value in sub_after.items():
		# 		count += 1
		# 		total_number[i].append(key)
		# 		if count > K:
		# 			break 
		# print(total_number)
		
		# pickle.dump(total_number,file1_object)
		# exit()

		#mean_rating_ = np.mean(validation[:, 2])
		# def predict(user_features_,item_features_, data,max_rating=5.0,min_rating=1.0):
		# 	if not mean_rating_:
		# 		raise NotFittedError()
		# 	u_features = user_features_.take(data.take(0, axis=1), axis=0)
		# 	i_features = item_features_.take(data.take(1, axis=1), axis=0)
		# 	preds = np.sum(u_features * i_features, 1) + mean_rating_
		# 	if max_rating:
		# 		preds[preds > max_rating] = max_rating
		# 	if min_rating:
		# 		preds[preds < min_rating] = min_rating
		# 	return preds
		# train_preds = bpmf.predict(ratings[:, :2])
		# train_rmse = RMSE(train_preds, ratings[:, 2])
		#bpmf.fit(validation, n_iters=eval_iters)
		# val_preds = bpmf.predict(validation[:, :2])
		# val_rmse += RMSE(val_preds, validation[:, 2])
		# print("after %d iteration, validation RMSE: %.6f" %
		#       (eval_iters,val_rmse))
	# else:
	# 	print(i)
	# val_rmse=val_rmse/5.0
	# print("after %d iteration, validation RMSE: %.6f" %
	# 	      (eval_iters,val_rmse))