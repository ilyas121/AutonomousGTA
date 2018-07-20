import numpy as np
from ../grabscreen import grab_screen
import cv2
import time
from keyBind import key_check
import os 

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

def outputKeys(keyList):
	output = [0,0,0,0,0,0,0,0,0]

	if 'W' in keys and 'A' in keys:
        	output = wa
	elif 'W' in keys and 'D' in keys:
        	output = wd
	elif 'S' in keys and 'A' in keys:
		output = sa
	elif 'S' in keys and 'D' in keys:
        	output = sd
	elif 'W' in keys:
        	output = w
	elif 'S' in keys:
        	output = s
	elif 'A' in keys:
        	output = a
	elif 'D' in keys:
        	output = d
	else:
        	output = nk
    	return output
def run(startnum, filename):
	filename = filename
	startnum = startnum
	training_data = []
	#Buffer to prepare yourself to drive
	for t in list(range(4))[::-1]:
        	print(t+1)
        	time.sleep(1)

    	prevTime = time.time()
    	paused = False
    	print('COLLECTING!!!')
	
	while(True):
		if not paused:
			view = grab_screen(region=(0,40,1920,1120))
			previousTime = time.time()
			#Take the view and resize to output 
			cnnView = cv2.resize(screen,(480, 270))
			cnnView = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
			
			output = outputKeys(key_check())
			training_data.append([cnnView, output])
			
			prevTime = time.time()
			
			if len(training_data) == 500:
				np.save(filename, training_data))
				print(str(startnum * 500) + "SAVED!!" )
				training_data = []
				startnum 
				
