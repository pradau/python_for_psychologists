#!/usr/bin/env python
#Assumptions
# - references to subject j means index j-1 in the Python array. e.g. Subject 2 has index 1.
# - references to face j means index j-1 in the Python array. e.g. Face 4 has index 3.

#Exercise D
import numpy as np
import scipy.io as sio
print("TEMP")
myscrewup

#1
#after loading 'data' array, you should set the type to float so that it doesn't default to integer operations.



print("#1")
print("stimno\n", stimno)
print("gender\n", gender)
print("symm\n", symm)
print("data after changes\n", data)

#2



print("#2")
print('mean {0:.3} min {1} max {2}'.format(mean, min, max))
for subject in range(data.shape[1]):
    print('range of attractiveness for subject {0} is {1}-{2}'.format(subject+1, np.min(data[:,subject]), np.max(data[:,subject]) ) )
#3



print("#3")
if rating_m > rating_w:
    print("Subject 1 rates men more attractive.")
elif rating_w > rating_m:
    print("Subject 1 rates women more attractive.")
else:
    print("Subject 1 rates both men and women equally attractive.")

#4



print("#4")
if rating_hisymm > rating_losymm:
    print("Subject 3 rates high symmetry more attractive.")
elif rating_losymm > rating_hisymm:
    print("Subject 3 rates low symmetry more attractive.")
else:
    print("Subject 3 rates both high and low symmetry equally attractive.")

#5



print("#5")
print("Agreement array\n", agree)
print('Number of agreements between subjects 2 and 3: ', agree_sum)

#6



print("#6")
#should take absolute value of disagreement before averaging.
print("Average disagreement {0:.3}".format(avg_disagree))

#7



print("#7")
print("Old attractiveness", avg_old)
print("Other attractiveness", avg_other)
print("It is " + str(avg_other > avg_old) + " that younger faces are more attractive than older in this experiment.")

#8
# Sorting is much easier with the relevant columns in a Pandas dataframe.
import pandas as pd



print("#8")
print("Dataframe with data and stimulus number\n",df)
