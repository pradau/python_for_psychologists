#!/usr/bin/env python
#Exercise H
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
#path to data files relative to notebook

#standard error function

#1
print("#1")

#### Scatter plot

# print(mat_contents)
err = mat_contents['err']
cong = mat_contents['cong']
stim = mat_contents['stim']
data = mat_contents['data']
# print(err)
print('cong')
print(cong)
# print(stim)
print('data')
print(data)




print("#2")


print('error\n',error)
print('notvalid\n',notvalid)
print('reaction times after excluding errors and those <200ms and >1000ms\n',goodrt)


#3
print("#3")
print("Overall exclusion rate for all conditions is:")

print("Exclusion rate for congruent condition is:")

print("Exclusion rate for neutral condition is:")

print("Exclusion rate for incongruent condition is:")


#4
print("#4")

#### Histograms
#arrangement of subplots


#number of histogram bins. Lower number is a more coarse histogram.






#5
print("#5")

print("results\n", results)

#6
print("#6")
#### Bar graph


#put in tuples for graphing







#7
# NOT completed.
