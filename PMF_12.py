# -*- coding: utf-8 -*-
import numpy as np

class PMF():
#num_user,num_item,train,test,0.01,10,0.01,0.01,100
    def train(self,num_user,num_item,U,V,train,test,learning_rate,K,regu_u,regu_i,maxiter):
        U=U
        print(U)
        V = V
        pre_rmse=100.0
        endure_count=3
        patience=0
        for iter in range(maxiter):#maxiter是迭代次数
            loss=0.0
            for data in train:
                #print(data)
                user=data[0]
                item=data[1]
                rating=data[2]

                predict_rating=np.dot(U[user],V[item].T)
                error=rating-predict_rating
                loss+=error**2
                U[user]+=learning_rate*(error*V[item]-regu_u*U[user])
                V[item]+=learning_rate*(error*U[user]-regu_i*V[item])

                loss+=regu_u*np.square(U[user]).sum()+regu_i*np.square(V[item]).sum()
            loss=0.5*loss
            rmse=self.eval_rmse(U,V,test)
            print('iter:%d loss:%.3f rmse:%.5f'%(iter,loss,rmse))
            if rmse<pre_rmse:   # early stop
                pre_rmse=rmse
                patience=0
            else:
                patience+=1
            if patience>=endure_count:
                break


    def eval_rmse(self,U,V,test):
        test_count=len(test)
        tmp_rmse=0.0
        for te in test:
            user=te[0]
            item=te[1]
            real_rating=te[2]
            predict_rating=np.dot(U[user],V[item].T)
            tmp_rmse+=np.square(real_rating-predict_rating)
        rmse=np.sqrt(tmp_rmse/test_count)
        return rmse