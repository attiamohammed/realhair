## This files generate the test images with hair mask
from tqdm import trange
import random,os
import os
import numpy as np
import skimage.io, skimage.transform
from skimage.morphology import dilation,disk,erosion
import matplotlib.pyplot as plt
import os,shutil, glob
import subprocess
import argparse
from skimage import img_as_ubyte


def hair_pair(i,j):
    image_nohair=images[i]
    image_nohair=skimage.transform.resize(image_nohair,(768,768))
    mask=masks[j]
    if len(mask.shape) > 2:
        mask=mask[:,:,1]
        mask=erosion(mask,disk(1))
    else:
        mask=dilation(mask,disk(4))
    mask=np.tile(mask[...,None],(1,1,3))
    mask=skimage.transform.resize(mask,(768,768))
    locs=np.where(mask>0.5)
    imageT=np.copy(image_nohair)
    imageT[locs]=1
    image_mask=np.hstack((imageT,mask))
    skimage.io.imsave(os.path.join(saveDir,images.files[i].split('/')[-1].split('.')[0]+'-'+masks.files[j].split('/')[-1].split('.')[0]+'.jpg'),img_as_ubyte(image_mask))
    skimage.io.imsave(os.path.join(imagesRoot,'nohair_test',images.files[i].split('/')[-1].split('.')[0]+'.jpg'),img_as_ubyte(image_nohair))
    skimage.io.imsave(os.path.join(imagesRoot,'mask_test',masks.files[j].split('/')[-1].split('.')[0]+'.jpg'),img_as_ubyte(mask))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help='Images Folder',default='./dataset')
    parser.add_argument('--o', help='Output Folder', default='./dataset/test')
    parser.add_argument('--mode', help='random or fixed', required=True)
    args = parser.parse_args()
    imagesRoot=args.i
    saveDir=args.o
    mode=args.mode
    print ('Input directory for images: %s'%imagesRoot)
    print ('Mode is: %s'%mode)
    if os.path.isdir(os.path.join(imagesRoot,'nohair_test')):
      shutil.rmtree(os.path.join(imagesRoot,'nohair_test'))
    os.mkdir(os.path.join(imagesRoot,'nohair_test'))
    if os.path.isdir(os.path.join(imagesRoot,'mask_test')):
      shutil.rmtree(os.path.join(imagesRoot,'mask_test'))
    os.mkdir(os.path.join(imagesRoot,'mask_test'))
    if os.path.isdir(saveDir):
      shutil.rmtree(saveDir)
    os.mkdir(saveDir)

    pbar=trange(100, desc='Bar desc', leave=True)
    images=skimage.io.ImageCollection(os.path.join(imagesRoot,'image/*.jpg'))
    masks=skimage.io.ImageCollection(os.path.join(imagesRoot,'mask/*.jpg'))
    j= random.randint(0,len(masks)-1)
    i= random.randint(0,len(images)-1)
    for i in pbar:
      if mode == 'random':
          j= random.randint(0,len(masks)-1)
          i= random.randint(0,len(images)-1)
      else:
          i= random.randint(0,len(images)-1)
      pbar.set_description(images.files[i].split('/')[-1]+"*"+masks.files[j].split('/')[-1])
      hair_pair(i,j)
      pbar.refresh()
    pbar.close()
