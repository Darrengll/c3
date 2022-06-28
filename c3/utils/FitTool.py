#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:19:32 2019

@author: dp
"""

from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# __all__ = ['fitdata','dampCosine','damping','fourier_cycle','skewed_lorentzian','qubit_spectrum',
#            'cavity_spectrum','pseudo_voige',]

class fitdata():
    def __init__(self,func,xdata,ydata,p0,bounds=None):
        self.func=func
        self.xdata=xdata
        self.ydata=ydata
        self.p0=p0
        self.p=''
        self.rmse=''
        self.y2=''
        if bounds is None:
            self.fit()
        else:
            self.fit_with_bound(bounds)

    def residuals(self,p,y,x):
        return y-self.func(x,*p)

    def RMSE(self,x,y1,y2):
        variances = list(map(lambda x,y : (x-y)**2, y1, y2))
        variance = np.sum(variances)
        rmse =  np.sqrt(variance/len(x))
        return rmse

    def fit(self):
        y=self.ydata
        x=self.xdata
        num=0
        plsq=leastsq(self.residuals,self.p0,args=(y,x))
        self.p=plsq[0]
        self.y2=self.func(x,*self.p)
        self.rmse=self.RMSE(x,y,self.y2)
        while (self.rmse>1e-3)&(num<1000):
            plsq=leastsq(self.residuals,self.p,args=(y,x))
            self.p=plsq[0]
            self.y2=self.func(x,*self.p)
            self.rmse=self.RMSE(x,y,self.y2)
            num+=1

    def fit_with_bound(self,bounds):
        '''

        :param bounds: (lb,ub)
        :return:
        '''
        y = self.ydata
        x = self.xdata
        popt,pocv= curve_fit(self.func,x,y,self.p0,bounds=bounds)
        self.p = popt
        self.y2 = self.func(x, *self.p)
        self.rmse = self.RMSE(x, y, self.y2)
        num = 0
        while (self.rmse>1e-3) and (num<1000):
            popt, pocv = curve_fit(self.func, x, y, self.p0, bounds=bounds)
            self.p=popt
            self.y2 = self.func(x, *self.p)
            self.rmse = self.RMSE(x, y, self.y2)
            num+=1


def damping(x,*params):
    amp,t1,offset=params
    y=amp*np.exp(-x/t1)+offset
    return y

def fourier_cycle(x, y):
    """
    FFT change to get function cycle t
    :return: t
    """
    x_transform = np.array(list(map(lambda a: a - x[0], x)))
    multiple = x_transform[-1] / 1  # 采样率倍率
    y_fourier = np.fft.fft(y)[range(int(len(y) / 2))]  # 傅里叶变换
    yf_real = y_fourier.real  # 实部
    yf_imag = y_fourier.imag  # 虚部
    y_modulus = np.sqrt(yf_real ** 2 + yf_imag ** 2)  # 求模数
    ym_incision = y_modulus[1:]
    index = np.argmax(ym_incision) + 1
    w = index / multiple * 2 * np.pi  # 求w
    t = (2 * np.pi) / w
    return t


def dampCosine(x, *p):
    T2, freq, phase, amp, e = p
    y = amp * np.exp(-x / T2) * np.cos(2 * np.pi * x * freq + phase) + e
    return y


def Cosine(x, *p):
    freq, phi, amp, offset = p
    y = amp * np.cos(freq*(x-phi)) + offset
    return y


def get_rabi_point(x, y1, y2):
    # version1
    # L = len(x)
    # if y1[int(L/4)]<y1[0]:
    #     freq0 = np.pi/(x[np.argmin(y1)]-x[0])
    #     F1 = fitdata(Cosine, x, y1,
    #                  [freq0 , np.pi - x[np.argmin(y1)]*freq0 , (max(y1) - min(y1)) / 2, (max(y1) + min(y1)) / 2])
    #     point1 = [x[np.argmin(F1.y2)],np.min(F1.y2)]
    #     F2 = fitdata(Cosine, x, y2,
    #                  [freq0 , np.pi*2 - x[np.argmax(y2)]*freq0 , (max(y2) - min(y2)) / 2, (max(y2) + min(y2)) / 2])
    #     point2 = [x[np.argmax(F2.y2)],np.max(F2.y2)]
    # else:
    #     freq0 = np.pi / (x[np.argmax(y1)] - x[0])
    #     F1 = fitdata(Cosine, x, y1,
    #                  [freq0, np.pi*2 - x[np.argmax(y1)] * freq0, (max(y1) - min(y1)) / 2, (max(y1) + min(y1)) / 2])
    #
    #     point1 = [x[np.argmax(F1.y2)],np.max(F1.y2)]
    #     F2 = fitdata(Cosine, x, y2,
    #                  [freq0, np.pi  - x[np.argmin(y2)] * freq0, (max(y2) - min(y2)) / 2, (max(y2) + min(y2)) / 2])
    #     point2 = [x[np.argmin(F2.y2)],np.min(F2.y2)]
    #
    #version2
    # lb = [-np.inf,x[0],-np.inf,-np.inf]
    # ub = [np.inf,x[-1],np.inf,np.inf]
    # bounds = (lb,ub)
    # freq1 = np.abs(np.pi/(x[np.argmin(y1)]-x[np.argmax(y1)]))
    # F1 = fitdata(Cosine, x, y1,
    #              [freq1, x[np.argmax(y1)], (max(y1) - min(y1)) / 2, (max(y1) + min(y1)) / 2], bounds=bounds)
    # point1 = [F1.p[1],F1.p[2]+F1.p[3]]
    # freq2 = np.abs(np.pi/(x[np.argmin(y2)]-x[np.argmax(y2)]))
    # F2 = fitdata(Cosine, x, y2,
    #              [freq2, x[np.argmax(y2)], (max(y2) - min(y2)) / 2, (max(y2) + min(y2)) / 2], bounds=bounds)
    # point2 = [F2.p[1], F2.p[2] + F2.p[3]]
    # # plot
    # fig,ax = plt.subplots(2,1)
    # ax[0].plot(x,y1,x,F1.y2)
    # ax[0].annotate("({:.5f},{})".format(point1[0],point1[1]),fontsize = 14,xy = point1)
    # ax[1].plot(x,y2,x,F2.y2)
    # ax[1].annotate("({:.5f},{})".format(point2[0],point2[1]),fontsize = 14,xy = point2)
    # fig.tight_layout()
    # fig.show()
    # fig.canvas.draw()
    #version3
    L = len(x)
    lb = [-np.inf, x[0], -np.inf, -np.inf]
    ub = [np.inf, x[-1], np.inf, np.inf]
    bounds = (lb, ub)
    if np.argmax(y1)<L/10 or (L-1)-np.argmax(y1)<L/10:
        y1 = -y1
        freq1 = np.abs(np.pi / (x[np.argmin(y1)] - x[np.argmax(y1)]))
        F1 = fitdata(Cosine, x, y1,
                     [freq1, x[np.argmax(y1)], (max(y1) - min(y1)) / 2, (max(y1) + min(y1)) / 2], bounds=bounds)
        point1 = [F1.p[1], -(F1.p[2] + F1.p[3])]
        F1.y2 = -F1.y2
        y1 = -y1
    else:
        freq1 = np.abs(np.pi/(x[np.argmin(y1)]-x[np.argmax(y1)]))
        F1 = fitdata(Cosine, x, y1,
                     [freq1, x[np.argmax(y1)], (max(y1) - min(y1)) / 2, (max(y1) + min(y1)) / 2], bounds=bounds)
        point1 = [F1.p[1],F1.p[2]+F1.p[3]]
    if np.argmax(y2)<L/10 or (L-1)-np.argmax(y2)<L/10:
        y2 = -y2
        freq2 = np.abs(np.pi / (x[np.argmin(y2)] - x[np.argmax(y2)]))
        F2 = fitdata(Cosine, x, y2,
                     [freq2, x[np.argmax(y2)], (max(y2) - min(y2)) / 2, (max(y2) + min(y2)) / 2], bounds=bounds)
        point2 = [F2.p[1], -(F2.p[2] + F2.p[3])]
        F2.y2 = -F2.y2
        y2 = -y2
    else:
        freq2 = np.abs(np.pi / (x[np.argmin(y2)] - x[np.argmax(y2)]))
        F2 = fitdata(Cosine, x, y2,
                     [freq2, x[np.argmax(y2)], (max(y2) - min(y2)) / 2, (max(y2) + min(y2)) / 2], bounds=bounds)
        point2 = [F2.p[1], F2.p[2] + F2.p[3]]
    # plot
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(x, y1, x, F1.y2)
    ax[0].annotate("({:.5f},{})".format(point1[0], point1[1]), fontsize=14, xy=point1)
    ax[1].plot(x, y2, x, F2.y2)
    ax[1].annotate("({:.5f},{})".format(point2[0], point2[1]), fontsize=14, xy=point2)
    fig.tight_layout()
    fig.show()
    fig.canvas.draw()
    return F1,F2

def skewed_lorentzian(x,*p):
    A1,A2,A3,A4,fr,Ql = p
    y = (A1+A2*(x-fr)+(A3+A4*(x-fr))/(1.+4.*Ql**2*((x-fr)/fr)**2))
    return y

def qubit_spectrum(flux,*args):
    f10max, fc, M, offset, d = args
    f10 = (f10max + fc)*np.sqrt(
        np.sqrt(1 + d**2 * np.tan(np.pi * M* (flux - offset))**2) * np.abs(np.cos(M * np.pi * (flux - offset)))) - fc
    return f10

def cavity_spectrum(x,*args):
    f10max, fc, M, offset, d, fr0, g = args
    delta = -fr0 + qubit_spectrum(x,f10max,fc,M,offset,d)
    kai = g**2/(delta*(1-delta/fc))
    fr = fr0 - kai
    return fr

def pseudo_voige(x,*args):
    a,b,c,d,e,f = args
    return a * ((1 - e) * np.exp(-(x - b) ** 2 / c ** 2) + e / (1 + (x - b) ** 2 / c ** 2)) + d + f * x

def gauss(x,*args):
    a,b,c,d = args
    return a*np.exp((x-b)**2/c**2)+d

if __name__=='__main__':
    def cosine(x,*p):
        a,b,c,d=p
        y=a*np.cos(2*np.pi*1e-3*x*b+c)+d
        return y
    xdata=np.linspace(0,np.pi*2,101)
    ydata=np.cos(xdata*10*2*np.pi*1e-3)*1/2+1/2
    p0=[1/2,10,0,1/2]
    F=fitdata(cosine,xdata,ydata,p0)
#    res=F.fit()