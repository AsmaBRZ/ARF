from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy 
from sklearn.metrics.pairwise import rbf_kernel
import random

def mse(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    return np.mean((np.dot(datax, w.T) - datay)**2)

def mse_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    return np.mean(2 * (np.dot(datax, w.T) - datay))

def hinge(datax,datay,w):
    """ retourn la moyenne de l'erreur hinge """
    return max(0,-datay * w.dot(datax))

def hinge_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur hinge """
    if np.mean(datay * np.dot(datax,w.T))<=0:
        return np.mean(-datay*datax)
    else:
        return np.zeros(w.shape)
def chunks(l, n):
    n = max(1, n)
    result=[]
    for i in range(0, len(l), n):
        result.append(l[i:i+n])
    return result

def add_bias(datax):
    new_dim = np.ones(datax.shape[0]).reshape(-1, 1)
    new_datax=np.hstack((datax, new_dim))
    return new_datax

#calculate the gaussian proj for an example x - miniX 
def gauss(x,datax, sigma):
    x_xdata=x - datax
    result=np.zeros((1,x_xdata.shape[0]))
    i=0
    for norm_s in x_xdata:
        result[0][i]=np.exp(-np.linalg.norm(norm_s, 2)**2 / (2. * sigma**2))
        i+=1
    return result

def projection(datax,kernel,sigma=0,miniX=[],datay=[]):
    if kernel=="bias":
        new_col=np.ones(len(datax))
        new_datax=np.hstack((datax, new_col))
        return new_datax
    if kernel=="poly":
        x1_2 = (datax[:, 0] * datax[:, 0]).reshape(-1, 1)
        x2_2 = (datax[:, 1] * datax[:, 1]).reshape(-1, 1)
        x1x2 = (datax[:, 0] * datax[:, 1]).reshape(-1, 1)
        new_datax=np.hstack((datax, x1_2, x2_2, x1x2))
        return new_datax
    elif kernel=="poly3D":
        x1_2 = (datax[:, 0] * datax[:, 0]).reshape(-1, 1)
        x2_2 = (datax[:, 1] * datax[:, 1]).reshape(-1, 1)
        x1x2 = (2**0.5*(datax[:, 0] * datax[:, 1])).reshape(-1, 1)
        new_datax=np.hstack((x1_2, x2_2, x1x2))
        return new_datax
    elif kernel=="gauss":
        new_datax=np.zeros((datax.shape[0],np.array(miniX).shape [0]))
        i=0
        for x in datax:
            new_datax[i:i+1,:]=(gauss(x,miniX,sigma))
            i+=1
        return new_datax
class Perceptron(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01, kernel="bias",sigma=0,typeUp="batch",bins=0,nbPoints=100,w_type='random'):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
            :kernel: noyau du perceptron
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        self.kernel = kernel
        self.sigma=sigma
        self.bins=bins
        self.typeUp=typeUp
        self.miniX=[]
        self.nbPoints=nbPoints
        self.w_type=w_type
        
    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        if(self.w_type=='random'):
            self.w = np.random.random((1,D))
        if(self.w_type=='mean'):
            self.w = np.array([datax.mean(0)])
        self.w_init=copy.deepcopy(self.w)
        self.w_hist=[]
        data_projected=copy.deepcopy(datax)
        
        if(self.kernel=='poly') or( self.kernel=='gauss') or (self.kernel=='poly3D') :
            if(self.kernel=='bias'):
                self.w = np.random.random((1,D+1))
                self.w_init=copy.deepcopy(self.w)
            if (self.kernel=='poly'):
                self.w = np.random.random((1,D+3))
                self.w_init=copy.deepcopy(self.w)
            elif (self.kernel=='gauss'):               
                #self.w = np.random.random((1,np.array(self.miniX).shape[0]))
                list_shuffle=np.arange(0,data_projected.shape[0],1)
                np.random.shuffle(list_shuffle)
                datax_s=[x for _,x in sorted(zip(list_shuffle,data_projected))]
                datay_s=[x for _,x in sorted(zip(list_shuffle,datay))]
                self.miniX=chunks(datax_s,self.nbPoints)[0]
                self.miniY=chunks(datay_s,self.nbPoints)[0]
                self.w = np.random.random((1,np.array(self.miniX).shape[0]))
                self.w_init=copy.deepcopy(self.w)                   
            elif (self.kernel=='poly3D'):
                self.w = np.random.random((1,D+1))
                self.w_init=copy.deepcopy(self.w)
            data_projected=self.project_data(copy.deepcopy(datax),datay)
        self.w_hist.append(self.w_init)
        if(self.typeUp=='batch'):      
            for i in range(self.max_iter):
                self.dd=data_projected
                self.cc=self.loss_g(data_projected, datay, self.w)
                for j in range(len(data_projected)):
                    self.w -= self.eps * self.loss_g(data_projected[i], datay, self.w)
                    self.w_hist.append(self.w)
        elif(self.typeUp=='stochastique'):
            for i in range(self.max_iter):
                index=random.choice(np.arange(0,data_projected.shape[0],1))
                self.w -= self.eps * self.loss_g(data_projected[index], datay[index], self.w)
                self.w_hist.append(self.w)
        elif(self.typeUp=='minibatch'):  
            list_shuffle=np.arange(0,data_projected.shape[0],1)
            np.random.shuffle(list_shuffle)
            datax_s=[x for _,x in sorted(zip(list_shuffle,data_projected))]
            datay_s=[x for _,x in sorted(zip(list_shuffle,datay))]
            mini_binsX=chunks(datax_s,1000)
            mini_binsY=chunks(datay_s,1000)
            for i in range(len(mini_binsX)):
                for j in range(self.max_iter):
                    self.w -= self.eps * self.loss_g(np.array(mini_binsX[i]), np.array(mini_binsY[i]), self.w)
                    self.w_hist.append(self.w)                   

    def predict(self,datax):
        data_projected=copy.deepcopy(datax)
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
        if(self.kernel=='poly') or( self.kernel=='gauss') or (self.kernel=='poly3D'):
            data_projected=self.project_data(copy.deepcopy(datax),None)
        return np.sign(np.dot(data_projected, self.w.T)).reshape(-1)

    def score(self,datax,datay):
        return np.where(self.predict(datax).reshape(-1)==datay,1,0).mean()
    def project_data(self,datax,datay):
        return projection(datax,self.kernel,sigma=self.sigma,miniX=self.miniX,datay=datay)    
