#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/8/13

@author: LLGuo
"""
import matplotlib.pyplot as plt
plt.style.use(['science','no-latex'])
def plotline(xlist,ylist_list,xname=None,yname=None,labellist=None):
    with plt.style.context(['science','no-latex']):
        fig, ax = plt.subplots()
        for i in range(len(ylist_list)):
            if labellist is not None:
                ax.plot(xlist, ylist_list[i],label=labellist[i])
            else:
                ax.plot(xlist, ylist_list[i])
#         ax.legend(title='Order')
        ax.set(xlabel=xname)
        ax.set(ylabel=yname)
        ax.autoscale()
        plt.legend()
#         fig.savefig('fig1.pdf')
#         fig.savefig('fig1.jpg', dpi=300)
    return fig,ax
def plot2D(xdata,ydata,zdata,xlabel=None,ylabel=None,title=None):
    with plt.style.context(['science','no-latex']):
        fig, ax = plt.subplots()
#         sc=ax.pcolormesh(xdata, ydata, zdata,cmap=plt.cm.jet)
        sc=ax.pcolormesh(xdata, ydata, zdata)
        ax.set(xlabel=xlabel)
        ax.set(ylabel=ylabel)
        ax.set(title=title)
        ax.autoscale()
        plt.colorbar(sc,ax=ax)
    return fig,ax
