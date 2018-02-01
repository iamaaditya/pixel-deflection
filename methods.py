from __future__ import division
from random import randint, uniform
import os, sys, argparse, matplotlib, multiprocessing
from keras.preprocessing import image
from scipy.misc import imread
from joblib import Parallel, delayed
from glob import glob
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
norm = matplotlib.colors.Normalize()


def denoiser(denoiser_name, img,sigma):
    from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means, wiener)
    if denoiser_name == 'wavelet':
        return denoise_wavelet(img,sigma=sigma, mode='soft', multichannel=True,convert2ycbcr=True, method='BayesShrink')
    elif denoiser_name == 'TVM':
        return denoise_tv_chambolle(img, multichannel=True)
    elif denoiser_name == 'bilateral':
        return denoise_bilateral(img, bins=1000, multichannel=True)
    elif denoiser_name == 'deconv':
        return wiener(img)
    elif denoiser_name == 'NLM':
        return denoise_nl_means(img, multichannel=True)
    else:
        raise Exception('Incorrect denoiser mentioned. Options: wavelet, TVM, bilateral, deconv, NLM')

def pixel_deflection(img, rcam_prob, deflections, window, sigma):
    H, W, C = img.shape
    while deflections > 0:
        for c in range(C):
            x,y = randint(0,H-1), randint(0,W-1)

            if uniform(0,1) > rcam_prob[x,y]:
                continue

            while True: #this is to ensure that PD pixel lies inside the image
                a,b = randint(-1*window,window), randint(-1*window,window)
                if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
            img[x,y,c] = img[x+a,y+b,c] 
            deflections -= 1
    return img
