#!/usr/bin/env python
#Exercise C
import numpy as np
import scipy.io as sio
path = '../matlab_exercises/matlab_tute/ex_C.mat'
#squeeze_me will turn the single column 2D matrix into a single row vector.
mat_contents = sio.loadmat( path, squeeze_me=True)
rt = mat_contents['rt']
cue = mat_contents['cue']
side = mat_contents['side']
print("rt\n",rt)
print("cue\n",cue)
print("side\n",side)

#1



print("#1")
print('valid\n',valid)

#2
# calc std with ddof=1. This is appropriate when you have a sample instead of entire population.



print("#2")
print('valid reaction times\n',rt[valid])
print('valid reaction time mean {0:.3} and sd {1:.3} msec'.format(rtmean, rtsd))

#3



print("#3")
print("error\n", error)

#4
# calc std with ddof=1. This is appropriate when you have a sample instead of entire population.



print("#4")
print("goodrt\n",goodrt)
print('good reaction time mean {0:.3} and sd {1:.3} msec'.format(goodrtmean, goodrtsd))

#5
#converts the integer array to logical array with left = True



print("#5")
print("side\n",side)

#6
#Compare the means for each side to provide a general idea. With more data you could do stat tests.



print("#6")

      
      
print('left vs. right reaction time means (all data): {0:.3} vs {1:.3}'.format(leftrtmean, rightrtmean))
print('left vs. right reaction time means (valid cues): {0:.3} vs {1:.3}'.format(valid_leftrtmean, valid_rightrtmean))

#for valid AND no error reaction times, we need all three conditions to find the reaction time arrays.
print('left vs. right reaction time means (valid cues and no errors): {0:.3} vs {1:.3}'.format(good_leftrtmean, good_rightrtmean))
