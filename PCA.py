import tensorflow as tf 
import numpy as np 
from numpy import *
from Optimizer import *

def pca(dataMat,topNfeat=9999999):
    #原始数据归一化
    meanVals=mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals
    #求协方差矩阵，python真方便，一大坨计算，一个函数搞定
    covMat=cov(meanRemoved,rowvar=0)
    #求特征值，特征向量
    eigVals,eigVects=linalg.eig(mat(covMat))
    #特征值排序
    eigValInd=argsort(eigVals)
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    redEigVects=eigVects[:,eigValInd]
    #原始数据映射到新的维度空间
    lowDataMat=meanRemoved*redEigVects
    reconMat=(lowDataMat*redEigVects.T)+meanVals
    return lowDataMat
