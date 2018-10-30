#!/usr/bin/env python
#Exercise G

import matplotlib.pyplot as plt
import numpy as np
# Using Pandas since we will sort linked arrays.
import pandas as pd
import scipy.io as sio
#path to data files relative to notebook


#standard error function


#1 & 2  Using 2 lists instead of a Matlab cell array
print("#1")

print("words\n", words)

print("#2")
#create frequency label list. It is meant to illustrate one way to do a larger list.



print("f_list\n",f_list)

#create a dictionary from the two lists, with assumption indices are matched.



# 3
print("#3")

#read the matlab file

#create a Pandas dataframe from the decision time data

#initialize the arrays so that we can do the append operations.


#loop through all of the words to create the average and standard error arrays.
# the arrays will have the order of the list 'words'



# print(df.columns)
# print(avg_freq_word)
# print(se_freq_word)

#### Bar graph
print("Bar graph")



#4
print("#4")

# Note we replace the 'switch' with if statement.



print("high_dt\n", high_dt)
print("med_dt\n", med_dt)
print("low_dt\n", low_dt)

# 5
print("#5")
print("Bar graph")
#### Bar graph



print(avg_freq)
print(se_freq)




#6
print("#6")
print("Saving output file in JSON format with name 'ExG6-Results.txt'")
# I'm not using Pandas DataFrames here because they are hard to use when the data doesn't fit
#  a well structured (rectangular array) form. They turn numpy arrays into strings.
# Instead I'll read / write JSON format files that are human readable and relatively standard.
# NOTE: this doesn't save all the data shown in this exercise, just some examples.
#  Each dictionary can have a different number of elements without restriction.

import json



#Create the dictionaries, one per word for each variable. The keys are constructed from the words
# so that the output file is more human-readable.
# Using 'word' as the key for all the dictionaries may make operations easier at the expense of
#  readability of the output file. In this case it is better to use a dictionary of dictionaries
#  instead of a list of dictionaries, with 'avg','se' and 'data' as the top level keys.


    #convert the data from Pandas series to list for compatibility with json.

#Prepare a list to contain the dictionaries in the desired order.



print('outdata')
print(outdata)

#Write the outdata list to file in JSON format

    #sort_keys, indent and separators fields make the output file easier to read.


#illustrate how to read from file and unpack the data. This isn't required by the question.
print("Bonus: Reading the data from file and unpacking some variables.")
#Read the list as read_data from file in JSON format


print('data')
print(read_data)
print('first dictionary, which has the average decision times.')

print('The average decision time for the word "coat"')

print('should match the result when calculating from the saved data in the third dictionary')
#numpy automatically converts the list array into a numpy array.
