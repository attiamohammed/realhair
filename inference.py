import torchvision.transforms as transforms
import time
import os
import torch
import functools
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
import util as util
from util import load_checkpoint,UnetGenerator,UnetSkipConnectionBlock
import functools
from PIL import Image
import torch.utils.data as data
import random
from skimage.io import imread,imsave
from skimage.morphology import dilation,disk
import shutil


loadSize=768
fineSize=768


image_path='./dataset/test'
images= os.listdir(image_path)

if os.path.isdir(os.path.join('dataset','simulated')):
        shutil.rmtree(os.path.join('dataset','simulated'))
os.mkdir(os.path.join('dataset','simulated'))
if os.path.isdir(os.path.join('dataset','fake')):
        shutil.rmtree(os.path.join('dataset','fake'))
os.mkdir(os.path.join('dataset','fake'))

for image in images:
        # Open input images in the folder "dataset/test"
        AB=Image.open(os.path.join(image_path,image)).convert('RGB')
        AB = AB.resize((loadSize * 2, loadSize), Image.BICUBIC)
        AB = transforms.ToTensor()(AB)

        # Split the images and resize to standard size
        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - fineSize - 1))
        h_offset = random.randint(0, max(0, h - fineSize - 1))
        # Extract the input mask with hair pixels are overlaid
        A = AB[:, h_offset:h_offset + fineSize,
                w_offset:w_offset + fineSize]
        B = AB[:, h_offset:h_offset + fineSize,
                w + w_offset:w + w_offset + fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        # Load the network and apply the trained model
        A = Variable(A, volatile=True)
        model2 = load_checkpoint('checkpoint.pth')
        output2=model2(A.unsqueeze(0))
        output2=util.tensor2im(output2.data)
        
        # Save output 
        image_pil = Image.fromarray(output2)
        image_pil.save(os.path.join('./dataset/simulated',image))

        image_h=imread(os.path.join('dataset/nohair_test',image.split('.')[0].split('-')[0]+'.jpg'))
        image_m=imread(os.path.join('dataset/mask_test',image.split('.')[0].split('-')[1]+'.jpg'))
        
        if len(image_m.shape)>2:
                image_m=image_m[:,:,1]

        image_m=dilation(image_m>np.max(image_m)/2,disk(3))
        image_m=np.tile(image_m[..., None],(1,1,3))
        hair_image=np.where(image_m>0)

        image_hair=np.copy(image_h)
        image_hair[hair_image]=output2[hair_image]
        imsave(os.path.join('./dataset/fake',image),image_hair)
