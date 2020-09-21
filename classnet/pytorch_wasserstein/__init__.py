import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import cv2

def gaussian(window_size, sigma):

    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window

def create_window_regionmae( window_size, channel ):

    window = torch.ones( [channel, 1, window_size, window_size] )
    return window

def _wasserstein( img1,img2,window,window_region_mae,window_size, channel, size_average = True):

    mu1 = F.conv2d( img1, window, padding = window_size//2, groups = channel )
    mu2 = F.conv2d( img2, window, padding = window_size//2, groups = channel )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq

    batchsize, C, height, width = mu1.size()

    mu1 = mu1.view( batchsize,C * height * width )
    mu2 = mu2.view( batchsize, C * height * width )

    sigma1_sq = sigma1_sq.view( batchsize,C * height * width )
    sigma2_sq = sigma2_sq.view( batchsize, C * height * width )

    wasserstein_map = ( F.pairwise_distance(mu1,mu2,2) + F.pairwise_distance(sigma1_sq,sigma2_sq,2) ) / 2

    mae = torch.abs(img1 - img2)
    region_mae = F.conv2d(mae, window_region_mae, padding=window_size // 2, groups=channel) / (window_size * window_size)

    if size_average:
        return wasserstein_map.mean() + region_mae.mean()
    else:
        return wasserstein_map.mean(1).mean(1).mean(1) + region_mae.mean(1).mean(1).mean(1)


class Wasserstein(torch.nn.Module):

    def __init__(self, window_size = 11, size_average = True):
        super(Wasserstein, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.window_regionmae = create_window_regionmae(window_size,self.channel)

    def forward(self, img1, img2):

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
            window_region_mae = self.window_regionmae

        else:
            window = create_window(self.window_size, channel)
            window_region_mae = create_window_regionmae(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
                window_region_mae = window_region_mae.cuda(img1.get_device())

            window = window.type_as(img1)
            window_region_mae = window_region_mae.type_as(img1)

            self.window = window
            self.channel = channel
            self.window_regionmae = window_region_mae

        return _wasserstein(img1,img2,window,window_region_mae,self.window_size,channel,self.size_average)

if __name__=='__main__':

    Wasserstein_loss = Wasserstein( window_size=11, size_average=True )

    img1 = Variable(torch.ones(1, 1, 11, 11))
    img2 = Variable(torch.zeros(1, 1, 11, 11))
    Wasserstein_loss( img1,img2 )
