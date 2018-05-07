#!/usr/bin/python
# -*- coding:utf-8 -*-
# power by Mr.Li
import torch as t
import  time

class BasicModule(t.nn.Moudule):
    '''
    主要提供save和load方法
    '''
    def __init__(self):
        super(BasicModule,self).__init__()
        self.moduel_name=str(type(self))
    def  load(self,path):
        self.load_stat_dict(t.load(path))
    def save(self,name=None):
        if name is None:
            prefix='checkpoints/'+self.moduel_name+"_"
            name=time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        t.save(self.stat_dict(),name)
        return name

class Falt(t.nn.Moudule):
    '''
    把输入reshape成（batch_size,dim_length）
    '''
     def __init__(self):
         super(Falt,self).__init__()
     def forward(self,x):
         return x.view(x.size(0),-1)