import numpy as np
from grabscreen import grab_screen
import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from collections import deque
from models import inception_v3 as neuralNet
from random import shuffle

FILE_I_END = 1860 

WIDTH = 480
HEIGHT = 270 
LR = 1e-3
EPOCHS = 30

netNAME = ''
prevNET = ''

loadNET = True

wl = 0
wdl = 0
sal = 0
sdl = 0
nkl = 0

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

net = neuralNet(WIDTH, HEIGHT, 3, LR, output=9, model_name=netNAME)

if loadNET:
    net.load(prevNET)
    print("Model loaded hopefully, LOLZ")

for e in range(EPOCHS):
    dataSequence = shuffle([i for i in range(1,FILE_I_END+1)])
    
    for epochCount, i in enumerate(dataSequence):
    
        try:
            filename = 'J:/phase10-random-padded/training_data-{}.npy'.format(i)
            train_data = np.load(filename)
            print('training_data-{}.npy'.format(i),len(train_data))
            
            train = train_data[:50]
            test = test_data[:50]
           
            test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
            test_y = [i[1] for i in test]
            
            model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
                snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)	

            if count%10 == 0:
               print("SAVING_MODEL!")
               net.save("MODEL_NAME")
        except Exception as e:
            print(str(e))

