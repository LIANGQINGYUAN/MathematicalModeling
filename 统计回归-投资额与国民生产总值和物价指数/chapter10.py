# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 19:49:36 2018

@author: liang
"""

import pandas as pd 
import numpy as np
import math
from sklearn.linear_model import LinearRegression as LR

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus']=False   #显示负号

'''
数据
'''
filename='./data_c10.xlsx'
df=pd.read_excel(filename)

x=df.iloc[:,2:].as_matrix()
y=df.iloc[:,1].as_matrix()


#要进行预测的下一个数据，即新的x值
x_new=np.array([3312,2.1938])
x_new=x_new.reshape((1,2))

#判断数据的线性相关性
plt.figure('x1_y')
plt.plot(x[:,0],y,'*')
plt.title('x1-y')

plt.figure('x2_y')
plt.plot(x[:,1],y,'*')
plt.title('x2-y')


'''
线性回归的基本训练

模型优点：R2＝0.9908，拟合度高

模型缺点：没有考虑时间序列数据的滞后性影响
        可能忽视了随机误差存在自相关；如果存在自相关性，用此模型会有不良后果
'''
lr=LR()
    
lr.fit(x,y)


#准确率
print('准确率：',lr.score(x,y))
#截距
print('截距：',lr.intercept_)
b_old=lr.intercept_
#系数
print('系数：',lr.coef_)
#计算训练集上的误差
y_predict=lr.predict(x)
print('训练集上的误差 : ',math.sqrt(np.mean((y[:]-y_predict[:])**2)))

#原始的模型
def model(x1,x2):
    return b_old+lr.coef_[0]*x1+lr.coef_[1]*x2


y_base=lr.predict(x_new)


'''
自相关性的定性诊断   残差诊断法

结果为大部分点落在第1, 3象限 ,所以存在正自相关性
'''
#计算残差的分布
e=y[:]-y_predict[:]
e_t=e[1:]   #t时刻的残差
e_t_1=e[:len(e)-1] #t-1时刻的残差
#画出散点图
plt.figure('e_distribute')
plt.plot(e_t,e_t_1,'.')

'''
自回归性的定量诊断   D-W检验 

1.计算DW值，判断有无自相关性
2.若有自相关性，则进行广义分差变换
'''
dw_old=sum((e_t-e_t_1)**2)/sum(e_t**2)
#dw_old=0.8754左右，根据样本容量20，变量数量3，95.5%的可信区间，查表可得dL=1.10, dU=1.54，存在正自相关性
p_hat=1-dw_old/2

#t和t-1时刻的x、y值
y_t=y[1:]
y_t_1=y[:len(y)-1]

x1=x[:,0]
x1_t=x1[1:]
x1_t_1=x1[:len(x1)-1]

x2=x[:,1]
x2_t=x2[1:]
x2_t_1=x2[:len(x2)-1]

#分差变换
y_star=y_t-p_hat*y_t_1
x1_star=x1_t-p_hat*x1_t_1
x2_star=x2_t-p_hat*x2_t_1

x_star=pd.DataFrame()
x_star[0]=x1_star
x_star[1]=x2_star
x_star=x_star.as_matrix()
#b_star=b(1-p_hat)
#产生新的模型为y_star=b_star+b1*x1_star+b2*x2_satr+u

###------------------------------------训练新的模型---------------------------------------###
'''
训练新的模型
'''
lr.fit(x_star,y_star)
#准确率
print('新的准确率：',lr.score(x_star,y_star))
#截距
print('新的截距：',lr.intercept_)
b=lr.intercept_
#系数
print('新的系数：',lr.coef_)


#要比较拟合图，需要先还原原始变量，因为新模型拟合的是y_star，而不是y
def newModel():
    return b+lr.coef_[0]*(x1_t-p_hat*x1_t_1)+lr.coef_[1]*(x2_t-p_hat*x2_t_1)+p_hat*y_t_1

#计算训练集上的误差
#y_predict_star=lr.predict(x_star)
print('新的训练集上的误差 : ',math.sqrt(np.mean((y[1:]-newModel())**2)))

#计算残差的分布
e_star=y[1:]-newModel()
e_t_star=e_star[1:]   #t时刻的残差
e_t_1_star=e_star[:len(e_star)-1] #t-1时刻的残差
#画出散点图
plt.plot(e_t_star,e_t_1_star,'*')
plt.title('残差分布  .:原模型   *：新模型')

'''
新模型的自相关诊断    D-W检验 
'''
dw_new=sum((e_t_star-e_t_1_star)**2)/sum(e_t_star**2)
#新的dw值为1.5751左右，样本容量n=19，回归变量数目k=3，=0.05 ，根据查表发现不存在相关性

'''
画图比较
'''
#1.残差图比较
plt.figure('e')
plt.plot(e,'.')
plt.plot(e_star,'*')
plt.title('残差图比较  .:原模型   *：新模型')
#2.拟合图比较
y_new_predict=newModel()

plt.figure('fit_compare')
plt.plot(y,'^-')
plt.plot(y_predict[1:],'.')
plt.plot(y_new_predict,'*')
plt.title('拟合图比较  .:原模型   *：新模型  ^ :原始数据')

#新数据的预测结果
y_new=b+lr.coef_[0]*(3312-p_hat*3073)+lr.coef_[1]*(2.1938-p_hat*2.0688)+p_hat*424.5

'''
预测下一次的结果
'''
print('基本模型的结果是：',y_base[0])
print('一阶自回归模型的结果是：',y_new)
