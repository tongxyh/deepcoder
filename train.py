import os

import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as f
import torch.utils.data as Data
import h5py
import rloss #missing

from torch import autograd
import multi_scale as MSM
from torchvision.models.vgg import vgg16
import torch_msssim as msssim
import argparse


parser = argparse.ArgumentParser(description='Deep Coder Train')
parser.add_argument('--save-path',default='models/0.2.2/',dest='save_path')

parser.add_argument('--psnr-thres',type=float,dest='psnr_t')
parser.add_argument('--num',type=int,dest='num')
parser.add_argument('--model',default=None,dest='model')

parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--train', dest='debug', action='store_false')
parser.set_defaults(feature=True)

args = parser.parse_args()

save_path = args.save_path
assert os.path.isdir(save_path), save_path+" model path don't exist"
print 'save to',save_path

device_ids = [1]
Res_Channel = 64
Mid_Channel = 5
epoch_begin = 0

NUM = args.num
BATCH_SIZE = 64
DEBUG = args.debug
RESTORE = False
if RESTORE == False:
    scale1_AE = MSM.Scale_1_AutoEncoder(Res_Channel=Res_Channel,Mid_Channel=Mid_Channel).cuda(device_ids[0])
else:
    restore_path = '/home/chentong/deepcoder/multi-scale-deepcoder/v-msssim/models/0.2.1/45_0.98987083_1.88811636.pkl'
    print 'restore model from', restore_path
    scale1_AE = torch.load(restore_path).cuda(device_ids[0])

#import os
#os.system('export CUDA_VISIBLE_DEVICES=%d'%(device_ids[0]))
if len(device_ids) > 1:
    print 'Using Multi GPU', device_ids
    scale1_AE = torch.nn.DataParallel(scale1_AE.cuda(device_ids[0]), device_ids=device_ids)
else:
    print 'Using GPU', device_ids[0]

FIXED_LAMBDA = True
if FIXED_LAMBDA == True:
    lambda_fix = 0.019
    print 'lambda is fixed to %0.8f'%(lambda_fix)
else:
    print 'lambda is augmented'

loss_func = nn.MSELoss().cuda(device_ids[0])
#entropy_func = rloss.rate_loss(device_id=1).cuda(device_ids[0])
mssim_func = msssim.MS_SSIM(max_val=1.0,device_id=1).cuda(device_ids[0])

# predict bit rates after ZPAQ
In_Dims = Mid_Channel *128/8*128/8
rate_estimator = MSM.Rate_PAQ(In_Dims=In_Dims).cuda(device_ids[0])
opt_R = torch.optim.SGD(rate_estimator.parameters(),lr=0.001)

opt_G = torch.optim.SGD(scale1_AE.parameters(),lr=0.0001,momentum=0.9)

print 'save epoch from:',epoch_begin
print 'fMap Channels:', Mid_Channel, ', ResNet Channels:', Res_Channel
print 'Batch Size:', BATCH_SIZE
print 'Is everything Correct?'

READY = raw_input()

if READY in ['y','Y','yes','Yes','YES']:
    file_train1 = h5py.File('/data/ImageCompression/train4.h5', 'r')
    train_data1 = torch.FloatTensor(file_train1['data'][:NUM]).permute(0,1,3,2)
    print train_data1.shape
    file_train2 = h5py.File('/data/ImageCompression/train5.h5', 'r')
    train_data2 = torch.FloatTensor(file_train2['data'][:NUM]).permute(0,1,3,2)
    print train_data2.shape
    file_train3 = h5py.File('/data/ImageCompression/train6.h5','r')
    train_data3 = torch.FloatTensor(file_train3['data'][:NUM]).permute(0,1,3,2)
    print train_data3.shape
    file_train4 = h5py.File('/data/ImageCompression/train7.h5','r')
    train_data4 = torch.FloatTensor(file_train4['data'][:NUM]).permute(0,1,3,2)
    print train_data4.shape
    file_train5 = h5py.File('/data/ImageCompression/train8.h5','r')
    train_data5 = torch.FloatTensor(file_train5['data'][:NUM]).permute(0,1,3,2)
    print train_data5.shape
    val1 = h5py.File('/data/ImageCompression/val1.h5','r')
    val_data1 = torch.FloatTensor(val1['data'][:NUM]).permute(0,1,3,2)
    val2 = h5py.File('/data/ImageCompression/val2.h5','r')
    val_data2 = torch.FloatTensor(val2['data'][:NUM]).permute(0,1,3,2)

    dataset1 = Data.TensorDataset(train_data1)
    dataset2 = Data.TensorDataset(train_data2)
    dataset3 = Data.TensorDataset(train_data3)
    dataset4 = Data.TensorDataset(train_data4)
    dataset5 = Data.TensorDataset(train_data5)

    dataset6 = Data.TensorDataset(val_data1)
    dataset7 = Data.TensorDataset(val_data2)

    loader1 = Data.DataLoader(torch.utils.data.ConcatDataset([dataset1,dataset2,dataset3,dataset4,dataset5]),
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             )

    epoches = (train_data1.shape[0]+train_data2.shape[0]+train_data3.shape[0]+train_data4.shape[0]+train_data5.shape[0])/BATCH_SIZE
    print "all %d epoches"%(epoches)

    loader2 = Data.DataLoader(torch.utils.data.ConcatDataset([dataset6,dataset7]),
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             )
    alpha = 0.0001

    for epoch in range(epoch_begin,400):
        entropy_avg = 0.
        mssim_avg = 0.
        val_loss = 0.
        if FIXED_LAMBDA:
            alpha = lambda_fix
        else:
            if epoch%5==0:
                alpha = alpha + 0.0001

        scale1_AE.train()

        for step, batch_x in enumerate(loader1):
            batch_x = Variable(batch_x[0]).cuda(device_ids[0])

            #split to encoder & decoder
            fake,x_entropy = scale1_AE(batch_x)

            r = rate_estimator(x_entropy)
            #print(x_entropy)
            paq_r = MSM.PAQ_encode(x_entropy).cuda(device_ids[0])
            paq_r = paq_r.view(paq_r.size(0), -1)
            #entropy_loss = entropy_func(x_entropy)
            entropy_loss = loss_func(r, paq_r)

            #mse_loss = loss_func(fake, batch_x)

            mssim_loss = 1.0 - mssim_func(fake,batch_x,levels=3)

            loss_sum = 1.0 * mssim_loss

            loss_re = alpha * entropy_loss

            #def show_grad(x):
            #    print torch.max(x).item(),torch.min(x).item()
            #x_entropy.register_hook(show_grad)
            #fake.register_hook(show_grad)

            #these two should work same
            scale1_AE.zero_grad()
            opt_G.zero_grad() #!!!
            loss_sum.backward()
            opt_G.step()

            rate_estimator.zero_grad()
            mssim_func.zero_grad()
            loss_re.backward()
            opt_R.step() #update rate estimator

            # TODO: loss_msssim & entropy_loss
            mssim_avg = mssim_avg + 1. - mssim_loss.item()
            entropy_avg = entropy_avg + entropy_loss.item()

            if step > 0 and step%1 == 0:
                # TODO: print avg
                print 'epoch:%d,step:%d,'%(epoch,step),'mssim:', mssim_avg/step,'entropy_loss:',entropy_avg/step,'lambda:',alpha
                #print x_entropy
                #print torch.histc(x_entropy.data.cpu(),63,0,63)

                # TODO: print time

        scale1_AE.eval()
        for step2, input_data in enumerate(loader2):
            with torch.no_grad():
                input_data = Variable(input_data[0]).cuda(device_ids[0])
                output,x_entropy2 = scale1_AE(input_data)
                #val_mse_loss1 = loss_func(output,input_data)
                val_mssim_loss1 = mssim_func(output,input_data)

                #TODO: avg val_entropy

                val_loss = val_loss + val_mssim_loss1.data.item()

        val_loss = val_loss / step2 # avg val_loss
        entropy_avg = entropy_avg / step # avg

        print '-----------------------------------'
        print 'epoch:',epoch,'eval_loss:',val_loss,'entropy_avg:',entropy_avg
        print '-----------------------------------'

        #DEBUG: save to default cuda device
        #TODO: Overwrite Warning
        if DEBUG:
            pass
        else:
            torch.save(scale1_AE.module, os.path.join(save_path,'%d_%.8f_%.8f.pkl' % (epoch, val_loss, entropy_avg)))
