#!/usr/bin/env python
#Exercise B

# Assumption 1: indexing
# It is assumed in the answers below that the descriptions that
# "row 1" means the first row i.e. 0th index in Python
# "element (2,2)" means the 2nd row, 2nd column in Python, i.e. indices 1,1.
# Assumption 2: cumulative effects
#  The changes in each line of within an Exercise may affect the answers in later parts of the exercise.
# Assumption 3: C and D are scalars, not "vectors" as suggested by question 1.

import numpy as np

#1




print('#1')
print('A\n', A)
print('B\n', B)
print('C\n', C)
print('D\n', D)
#2
#for F, multiply each element of B by scalar D
#for G, element-wise multiplication
#for J, divide each element of B by scalar D
print('#2')
print('E\n', E)
print('F\n', F)
print('G\n', G)
print('H\n', H)
print('J\n', J)

#3



K = np.concatenate((A,B), axis=0)
print('#3')
print('M\n', M)
print('N\n', N)
print('out = M+N\n', out)
print('A\n', A)
print('sum\n', sum)
print('K\n', K)

# 4



print("#4")
print("matrix multiplication of A by B transpose")
print("element wise multiplication A*B")

#5



print("#5")
print("max of each column of J in an array")
print(Jmax)
print("min of each row of B in an array")
print(Bmin)