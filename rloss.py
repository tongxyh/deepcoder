import torch
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as f
import torch.utils.data as Data
import h5py
#from ConvLSTM import *
import matplotlib.pyplot as plt
import torch.autograd as autograd
import numpy as np

#Treat it as a Class
class rate_loss(nn.Module):
    # rate_loss for pytorch-0.4.0
    # Written by Tong Chen at 2018.6.29
    def __init__(self, bins_per_unit=5,device_id=0):
        super(rate_loss, self).__init__()
        self.bins_per_unit = bins_per_unit

        filters = torch.FloatTensor([1. for i in range(self.bins_per_unit)])
        filters.resize_(1,1,self.bins_per_unit) #[out_channels, in_channels, Width]
        self.filters_sum = nn.Parameter(filters)

        filters = torch.FloatTensor([-1.,1.])
        filters.resize_((1,1,2))
        self.filters_1 = nn.Parameter(filters)
        self.device_id = device_id

    def forward(self,x):
        with torch.no_grad(): # modified for torch-0.4.0
            input = x.data.cpu()

            vmin = torch.floor(torch.min(x)) - 1
            vmax = torch.ceil(torch.max(x)) + 1

            nbins = torch.Tensor.int((vmax.data.cpu() - vmin.data.cpu())*self.bins_per_unit)
            hist = torch.histc(input,nbins,vmin.data,vmax.data)
            hist.resize_((1,1,nbins)) # N,C,W

            ele_sum = torch.sum(hist).item()
            hist = hist/ele_sum

            hist_add = torch.nn.functional.conv1d(hist.cuda(self.device_id),self.filters_sum,stride=1,padding=0)

            vmin_new = vmin + 0.5
            vmax_new = vmax - 0.5
            g = torch.div(torch.nn.functional.conv1d(hist_add, self.filters_1, stride=1, padding=0) , 1.0/self.bins_per_unit)

            #TODO: is g_constrain necessary
            #print g

            interval = 1./self.bins_per_unit

        def body(i,x):
            rloss = 0.
            mask = torch.lt(x, vmin_new + (i+1.) * interval) * torch.ge(x,vmin_new + i * interval)
            d = torch.masked_select(x,mask)

            if d.size()[0] > 0: # dismiss zero
                nloss = (d - (vmin_new+i*interval)) * g[0,0,i]  + hist_add[0,0,i] + 0.00000001
                log_rloss = torch.log(nloss) / -torch.log(2.0)
                rloss = torch.sum(log_rloss)
            return rloss

        rloss = 0.0
        for i in range(g.size()[2]):
            rloss = body(i,x) + rloss

        # TODO: cuda() outside the function

        return rloss/ele_sum
