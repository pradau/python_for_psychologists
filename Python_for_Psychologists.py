#!/usr/bin/env python
# coding: utf-8

# # Python for Psychologists

# This notebook follows the outline of "Matlab for Psychologists" with translation of the code into Python, and more context is provided in that article.
# A recommended reference is the [Python for Beginners](https://www.pythonforbeginners.com) website.

# ## Getting started

# These notes assume that you are either using MacOS / Linux, or Windows with [Powershell](https://docs.microsoft.com/en-us/powershell/scripting/powershell-scripting?view=powershell-6) for command line operations.
# I recommend installing the [Anaconda](https://www.anaconda.com/download) distribution of Python version 3.x that is available for most major platforms. This should be done even if you already have a version of Python on your computer. e.g. MacOS has Python built-in but it is the older version 2.7.

# The page [Python Setup](https://www.pythonforbeginners.com/basics/python-setup/) describes the basics of getting started.
# However, ignore the installation section of that page as you will use Anaconda instead.

# If you are not familiar with the command line interface (CLI) to your computer, i.e. the terminal window, then you should take some time to read a tutorial. Here is such a [tutorial for MacOS terminal](http://blog.teamtreehouse.com/introduction-to-the-mac-os-x-command-line).

# After installing Anaconda and re-booting your computer, your terminal should be correctly setup to find Python 3.x (currently version 3.6). Try the following command in a terminal to confirm your version.<br>
# **python -V**<br>
# The result should look like the following line, though you will likely have a newer Python version.<br>
# **Python 3.5.4 :: Anaconda custom (64-bit)**

# To check the python version in code you would do the following. The details are not important right now, but you should find the code runs without an error or else your Python version is too low.

# In[1]:


import sys
print(sys.version_info)
# the major and minor version numbers are printed
print(sys.version_info[0],sys.version_info[1])
#if the following statement causes an error then your version is lower than v.3.0
assert sys.version_info >= (3,0)


# To follow this tutorial you will be trying Python commands interactively. Although you can test short code segments interactively in the terminal using ipython, I recommend instead that you use Jupyter notebook which is feature rich and permits longer code chunks.  This program is already installed with your Anaconda installation. To use it with this notebook you should open a terminal and change directory to the one where the notebook file is located. e.g. If your Python course files are in a folder called "IntroToPython" on your desktop then type the following command to change your terminal directory accordingly<br>
# **cd ~/Desktop/IntroToPython**<br>
# and you can test that it is available by typing the following in the terminal<br>
# **jupyter notebook**<br>
# which should start the interface in your default web browser. You will see the directory contents which should include the notebook file "Python_for_Psychologists.ipynb". Double-click to open it in your browser.<br>
# Please read a good [tutorial for using Jupyter notebook](https://www.dataquest.io/blog/jupyter-notebook-tutorial/).
# When you have a Jupyter notebook open you can insert new code blocks and run them interactively. To insert comments you can start a line with the "#" comment symbol or else create a Markdown block such as this one you're reading which provides more ways to beautify your text.<br>
# **I'll refer to Jupyter notebook as JN for brevity.**

# ## Getting help

# To get help at anytime in a jupyter notebook you can precede a command name with the "?" symbol. This will show the help in a separate pane at the bottom of your notebook page
# Run the next code block to see how to get help for the len() function.

# In[2]:


# get_ipython().run_line_magic('pinfo', 'len')


# In[3]:


#the same but differently formatted info from help(<function>)
# help(len)


# # Lesson 1 - The Basics

# In[4]:


A = 10
A+A
B = 5+8
B


# JN will print the result of a simple statement like a 'A+A' or contents of a variable 'B' if listed on a separate line at the end of the code block. But it is good practice to always use the **print** function instead, as it is required in Python scripts and this is the normal way to use Python.

# In[5]:


print(A+A)
print(B)


# It is also very convenient to include a label with your result. Do this by combining it with a string.

# In[6]:


A = 15
print('B', B)
print('A\n', A)


# To clear a variable from memory (and not just set it to zero) then use **del**. This would be rarely used in small scripts but could be useful to remove a large array of numbers from memory.<br>
# There is no analogue to "clear all" in Python but if you need to do this in your notebook then use the Jupyter Notebook Kernel menu. e.g. Kernel / Restart and Clear Output to restore to the initial state.
# Or else use the **%reset** command as shown below but this is only used with ipython or Jupyter notebook.

# In[7]:


B=1
del B
#The next statement to display B causes an error because B is no longer defined.
#print(B)


# To see variables in the workspace<br>
#     **dir()** dictionary of in-scope variables:<br>
#     **globals()** dictionary of global variables<br>
#     **locals()** dictionary of local variables
#

# In[8]:


#Get a list of variables in the local scope (which includes hidden variables from your jupyter notebook)
dir()


# In[9]:


#this will clear the user-defined variables, when using JN. I've commented it out because it stops execution
#  of the following lines.
# %reset


# In[10]:


#This can be used to filter dir() to get the variables of the most interest to the typical user of JN. Don't worry about the details for now.
varlist = [x for x in dir() if '_' not in x]
print(varlist)


# # Lesson 2 - Matrices and Punctuation

# In general, there are several ways to manipulate a set of data that could represent a vector or matrix. I recommend:<br>
#  - if there is only a need to do simple operations like **sum** then a built-in *list* might suffice.<br>
#  - if matrix/vector multiplication will be required then use a Numpy *array*.<br>
#  - if the data is mixed with non-numeric then start with a Pandas *dataframe* from which you can extract *arrays*.<br>
#
# For this lesson, we'll stick to Numpy operations and objects.<br>
# Here's a good [cheat sheet for Numpy operations](https://www.dataquest.io/blog/numpy-cheat-sheet/).
# And here's a guide to [Numpy for Matlab users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html).
#

# In[11]:


#obtain the external module numpy whose commands we will specify by name (alias) np
import numpy as np
#Scalar array
A = np.array(10)
#Vector array
B = np.array([1,2,3])
print('A')
print(A)
print('B')
print(B)
B  #If you list a variable on the final like this in JN then you will see it shown as an Output object (e.g. array([1, 2, 3]))


# We can use reshape() to create a column vector. But this will also convert to a 2D array which is denoted by double brackets [[...]]

# In[12]:


#create row vector then reshape to be a column vector
C = np.array([4,3,8])
C = C.reshape(-1,1) #-1 indicates number of rows is inferred from specified number of columns (1)
print('C')
print(C)


# In[13]:


#Create a matrix
D = np.array([[5,6,7,9],
              [8,3,5,3],
              [5,6,3,2]])
print(D)


# Conveniently, Python will usually take care of variable type conversion for you *without* explicitly setting the type (i.e. implicit conversion). This next example shows that although the print statements indicate the same number, in fact there is a type difference.

# In[14]:


#Element
#convert an array of size 1 (scalar array) into an ordinary scalar element
print(A)
print(type(A)) #before conversion it is a numpy array
a = np.asscalar(A)
print(a)
print(type(a)) #after conversion it is an int (integer)


# A common example of how this could matter is when trying to print a number as part of a sentence. If we try to mix the string (in "") with the integer without type conversion then you can get an error.

# In[15]:


# This next line produces the type conversion error: "TypeError: Can't convert 'int' object to str implicitly"
# print("Your score is " + a + "!!!")
# This next line fixes it by converting the type( from integer to string) so that the strings can be combined by '+' into one string for the print().
print("Your #4 score is " + str(a) + "!!!")


# ## Brackets

# As shown above you can enter data by initializing np.array() with a list in square brackets [].
# There is more than one method to create a column vector, one shown above and another below.

# In[16]:


D = np.array([[3],[1],[6],[5]]) #create a 4x1 matrix
print(D)


# To create a 2D matrix you can use a nested list for initialization, one list in \[...] for each row. To access row 2, column 3 of E then index it by using the indices.<br>
# **Important** Unlike Matlab, Python starts indices from 0 (not 1) so row 2 is indexed by 1, column 3 by index 2 and in general row *m* by *m-1* and column *n* by *n-1*.

# In[17]:


E = np.array([[1,2,3],[4,5,6]]) #create a 2x3 matrix
print(E)
print(E[1][2])


# In[18]:


#this produces an indexing error
#print(E[3][4])


# To change part of a matrix you can also use square brackets. e.g. E[][] or more compactly E[,]

# In[19]:


E[0][2]= 10
print('first\n',E)
# a more compact way to access the same element is using a "," instead of the interior brackets ']['
E[0,2] = 10
print('2nd\n',E)


# Add a row to the *E* matrix. Here we use the *axis* argument which is often used in numpy to specify the direction of operations. I like to think of *axis=0* as "down the rows" and *axis=1* as "across the columns." In this case we concatenate *down* and add a row. Without using an axis argument the direction is determined implicitly.

# In[20]:


#add a row.
E = np.concatenate((E, [[7,8,9]]), axis=0)
print(E)


# In[21]:


#demonstrates the difference between axis=0 and axis=1
AX = np.array([[1,2,3],[4,5,6]])
print(AX)
print('sum down', np.sum(AX, axis=0))
print('sum across', np.sum(AX, axis=1))


# ## Slicing (colon indexing)
# You can read about this in the [Numpy reference](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html).

# In[22]:


#here the : means "every row" because it is before the comma, and 1 indexes the 2nd column.
print(E[:,1])


# In[23]:


#here the : means "every column" because it is after the comma, and 1 indexes the 2nd row.
print(E[1,:])


# We can use the range() function to obtain a list of numbers. We have to be careful to remember that it uses **end+1** as 2nd argument to indicate stopping value. In this example the numbers will be [5,10] inclusive.

# In[24]:


#we get a list of consecutive integers by range() and initialize an array with it
F=np.array(range(5,11))
print(F)


# In[25]:


F=np.array(range(5,11))
#convert to a 2D matrix with 1 row and as many columns as the original size of F
F = F.reshape(1,F.size)
print(F)
# everything in columns 3, 4 and 5 of F. NOTE that using a colon in the list [] to specify the columns is not allowed.
F2 = F[:,[2,3,4]]
print(F2)
# here's another way to get the list specifying the columns
cols = range(2,5)
print(F[:,cols])


# In[26]:


#use a step for counting when creating a consecutive integer list from 3 to 20, in steps of 4.
# NOTE that range does NOT use colons and the format is (start, end+1, step)
G = np.array(range(3, 21, 4))
print(G)


# Python has no analogue to the "Array Editor" of Matlab.

# # Lesson 3 - Indexing
# There is no builtin function in Python to create a magic square so we will input it from scratch.

# In[27]:


A=np.array([[17,24,1,8,15],
            [23,5,7,14,16],
            [4,6,13,20,22],
            [10,12,19,21,3],
            [11,18,25,2,9]])
print(A)


# In[28]:


#store 4th row, 3rd column element of A in B
B=A[3,2]
print(B)


# In[29]:


#store 4th row of A in C
C=A[3,:]
print(C)


# In[30]:


#extract 2nd and 3rd row of A and store in D
D=A[[1,2],:]
print(D)


# In[31]:


#extract columns 1 to 3 of D, for all rows.
E=D[:,[0,1,2]]
print(E)


# In[32]:


#alternative, compact way to extract a range of columns using the list built from range() of consecutive numbers
# building a list explicitly as follows is not necessary but can help with debugging because you can see the list.
# print([x for x in range(0,3)])
E=D[:,range(0,3)]
print(E)


# In[33]:


#set the 3rd row,3rd column of A to 100.
A[2,2]=100
print(A)


# In[34]:


#set the 4th column to 0
A[:,3]=0
print(A)


# ** *You should know enough now to do Exercise A. * **

# ## Lesson 4 - Basic math
# Some vector/matrix can be accomplished with concise operators (like + or -) while others require a Numpy function.

# In[35]:


#vector addition
A=np.array([1,2])
B=np.array([5,6])
C = A + B
print(C)


# In general to make a 2D matrix from a 1D vector, we need to first create the 2D matrix. We do this by putting the 1D array in square brackets.

# In[36]:


C = np.array([C])
print(C)
#transpose to get a column vector from the original row vector.
print('transpose\n',C.transpose())
# Note that you can do this even if C was a row vector instead of a 2D matrix.
print('A more compact way to transpose where T is an alias to the transpose function')
# The .T is an accessor to the transpose function.
print(C.T)
C = np.array([6,8])
#matrix multiplication. In this case it is same as dot product.
print(np.matmul(C,C.T))


# A more typical example is where we want to multiply together two 2D matrices. Here is an example that demonstrates how transpose might be used.

# In[37]:


E = np.array([[2,2],[1, 1]])
print(E)
F = np.array([3,4,5,6,7,8])
#create a 3x2 matrix from the vector.
F=F.reshape(3,2)
#this line will fail as 'shapes not aligned' because E is 2x2 and F is 3x2
#print(np.matmul(E,F))
# this line succeeds because E (2x2) can be multiplied with F.T (2x3)
print(np.matmul(E,F.T))


# **np.matmul** for matrix multiplication<br>
# **np.multiply** for element-wise multiplication<br>
# **np.dot** for dot product of two arrays<br>
# **np.divide** for element-wise division<br>
# **np.power** for raising a matrix to a power<br>
#

# In[38]:


#simple math doesn't require numpy
print(174*734)


# In[39]:


#simple math with a scalar variable. NOTE that A is not changed here.
A=10
print((A*2)/5)


# In[40]:


#2D matrix is multiplied elementwise by the scalar A. The 2nd line and indent is for clarity, not required.
E=np.array([[1,2,3],
            [4,5,6]])
J= E*A
print(J)


# In[41]:


#matrix subtraction
K = J - E
print(K)


# In[42]:


#create a 2D matrix
L=np.array([[3,2,1],[9,5,6]])
print(L)


# In[43]:


#element-wise division
print(K/L)


# In[44]:


#element-wise multiplication of two matrices with same shape.
print(E*L)


# ## Lesson 5 - Basic functions
# Functions can be used in Python similarly to Matlab.

# In[45]:


#sum the columns of K. axis=0 means the vertical direction, axis=1 means horizontal direction
print(np.sum(K,axis=0))


# In[46]:


#sum the rows of K.
print(np.sum(K,axis=1))


# In[47]:


#find the mean of J (for each column) and store in mj
mj = np.mean(J, axis=0)
print(mj)


# In[48]:


#transpose of K
print(K.T)


# Note that you can construct a matrix from a row vector using **reshape**. Notice that it fills in the first row from the first set of elements in the input, and continues filling it in row-wise.

# In[49]:


#create a matrix of random numbers with 3 rows and 5 columns
#random is submodule of numpy. Here the function gives 15 element vector with values in [0,1]
rows = 3
cols = 5
R = np.random.rand(rows*cols)
print('row vector')
print(R)
#reshape from row vector to 3x5 matrix.Here the 'rows' variable could be replaced by -1 and numpy will calculate it.
R = np.reshape(R, (rows,cols))
print('3x5 matrix')
print(R)
#reshape will error if you input the wrong number of elements to form the matrix you tell it to make.
#R = np.reshape(R, (3,6))


# ** *You should know enough now to do Exercise B. * **

# ## Lesson 6 - Logical Operators
# Most of the logical operators are the same in Python and Matlab. A difference is that 'not' is indicated by the exclamation point (!) rather than tilde (~).<br>
# - < Greater than
# - < Less than
# - == Equal
# - != Not equal
# - & AND
# - | OR
# - ! NOT
# <br>
# In Python, we can apply boolean operations to arrays and the result is a Boolean array having elements of *True* or *False*. These are treated as 1 or 0 respectively for most operations.

# In[50]:


A = np.array([1,5,3,4,8,3])
B = A>2
print("B shows which elements of A are greater than 2\n",B)
print("An element of B is type:", type(B[0]))  #numpy boolean
print("An ordinary Boolean in Python (not numpy) is a separate type but usually can be interchanged with a numpy bool by implicit conversion.")
test_bool = True  #Python boolean
print(type(test_bool))
test = [False, True,True,True,True,True]  #list from Python booleans
print("Are the 2 boolean arrays equal?\n", B == test)  #check the equality of the numpy array and list


# In[51]:


C = A<5 #is less than
print(C)


# In[52]:


D = B & C # AND
print(D)


# In[53]:


#extract the array from A where the boolean array D was True (i.e. condition D was met)
E = A[D]
print(E)


# In[54]:


#can do this extraction in one line without the intermediate arrays.
print(A[(A>2) & (A<5)])  #Array composed of elements of A that are greater than 2 AND less than 5


# In[55]:


#convert a logical index (D) into a subscript index (F). D is a boolean mask, where 3rd, 4th and 6th values of D are True.
F=np.where(D)
print(D)
print(F)


# In[56]:


#These indices can be used to extract the array from A meeting condition D (same result as above)
print(A[F])


# In[57]:


#some data that corresponds to each cat. e.g. age
data = np.array([4,14,6,11,3,14,8,17,17,12,10,18])
#type of cat (of 3 types)
cat = np.array([1,3,2,1,2,2,3,1,3,2,3,1])


# In[58]:


#the parentheses are for clarity only. Which cats are type 2?
cat2 = (cat == 2)
print(cat2)


# In[59]:


#show the data (age) of type 2 cats.
data2 = data[cat2]
print(data2)


# In[60]:


#find the mean of the data of type 2 cats. NOTE np.average() is equivalent to np.mean()
print(np.mean(data2))


# In[61]:


#and here is the same calculation in compact form.
print(np.average(data[cat==2]))


# ** *You should know enough now to do Exercise C and D. * **

# ## Lesson 7 - Missing Data
# Missing data (NaNs) in Python are represented with np.nan. There are also a variety of calculators which will ignore missing data including:<br>
# - np.nansum()
# - np.nanmean()
# - np.nanvar()
# - np.nanstd()

# In[62]:


print(np.nan)


# In[63]:


#dtype converts this from a integer to boolean array
err = np.array([1,0,0,0,0,0,0,1,0,1,0,0],dtype=bool)
print(err)
#Let's set all of the data with errors to np.nan so that they will be ignored in subsequent calc
# first convert the data to float type which is compatible with np.nan insertion (unlike integer arrays)
newdat = np.array(data,dtype=float)
print(newdat)
# insert the np.nan where the errors are according to mask err.
newdat[err==True] = np.nan
print(newdat)


# In[64]:


#Now the extracted data of cat type 2 will have NaN in it
data2 = newdat[cat2]
print(data2)


# In[65]:


#If use the ordinary mean() we get "nan" meaning "can't calculate on this array"
print(np.mean(data2))


# In[66]:


#Therefore we use the nanmean() instead
print(np.nanmean(data2))


# ## Lesson 8 - Basic Graphs
# There are multiple graphing options available in Python. The most common 2D plotting option is Matplotlib, and it is also the most similar to Matlab plot().
#

# In[67]:


#sequence of 30 numbers from 0 to 2PI, i.e. the radian argument
x = np.linspace(0, 2*np.pi, num=30)
print(x)


# In[68]:


#corresponding array of sin() values.
y = np.sin(x)
print(y)


# In[69]:


z = 0.5 + np.cos(x/2)
print(z)


# In[70]:


import matplotlib.pyplot as plt
# Using "tab:..." for colors gives alternate colors that are easier on the eyes.
#plt.plot(x, y, 'b-o', color='tab:blue')
#plt.plot(x, z, 'r--', color='tab:red')
# NOTE that the marker conventions are nearly identical.
plt.plot(x, y, 'b-o')
plt.plot(x, z, 'r--')
plt.xlabel('x')
plt.title('a couple of lines')
#note that the legend items need to be in a list
plt.legend(['y=sin(x)','z=0.5+cos(x/2)'])
# Save figure to file.
plt.savefig('Figure.png')  #In JN, must save your figure before using the show() command
#The next line creates a interactive popup if it is in your script (instead of JN where it inserts it in the notebook)
plt.show()
plt.close()  #good form to close (clean up) at end otherwise you might have bad effects on following figures.


# ### Plot some data
# Make sure the file lesson2.mat is in a subdirectory of your current directory called 'matlab_exercises' and then run the following. You should see a 2D matrix with 3 columns & 10 rows.

# In[71]:


import scipy.io as sio
import os
import matplotlib.pyplot as plt
import numpy as np

#squeeze_me will turn the single column 2D matrix into a single row vector.
mat_contents = sio.loadmat( 'matlab_exercises' + os.sep + 'lesson2.mat', squeeze_me=True)
data = mat_contents['data']
print(data)


# To first look at all of the data, we could try a simple plot. The row index will be *x* and data values will be *y*. If we use marker *bo* we will see only points, whereas with *b-o* we'll see separated lines for each column.

# In[72]:


plt.plot(data, 'bo')


# We could look for a trend by plotting a regression line. Let's assume the columns are repeated trials and the rows are discrete conditions (like birth year), so we average across columns.

# In[73]:


#obtain the fit parameters from linear regression
# x will be the row indices
x = range(data.shape[0])
print(x)
# for each row, find average across columns to make a 1D vector.
avg = np.average(data, axis=1)
print(avg)
fit = np.polyfit(x, avg, 1)
#obtain the function that represents the fit and can generate new values.
fit_fn = np.poly1d(fit)
#plot fit as red line with triangles.
plt.plot(x, fit_fn(x), 'r-^')


# Let's improve this with error bars, in this case standard deviations across each row of data.

# In[74]:


err = np.std(data, axis=1)
print(err)
plt.errorbar(x, fit_fn(x), yerr=err, fmt='g-')


# Now let's put all of this together and plot on a single graph.
# **Warning**: It's a quirk of JN that you must put all the plotting commands in a single cell in order to see them overlaid on one plot.

# In[75]:


plt.plot(data, 'bo')
x = np.arange(data.shape[0])
avg = np.average(data, axis=1)
fit = np.polyfit(x, avg, 1)
fit_fn = np.poly1d(fit)
plt.plot(x, fit_fn(x), 'r-^')
err = np.std(data, axis=1)
plt.errorbar(x, fit_fn(x), yerr=err, fmt='g-')
plt.savefig('Figure-Lesson8.png')
plt.show()
plt.close()


# Plot a new figure with subplots that will be called Figure 2.

# In[76]:


#arrangement of subplots
nrows = 2
ncols = 1
idx = 1
#row 1 subplot
plt.subplot(nrows, ncols, idx)
plt.plot(data,'b*')
plt.title('Raw data')

#row 2 subplot
plt.subplot(nrows, ncols, idx+1)
plt.bar(x, fit_fn(x), color='r')
plt.show()
plt.close()


# Now repeat this except with error bars (standard deviation) in color green with star mid-point markers. The x-axis limits are also set so that the lower bars line up with the points in the upper plot.

# In[77]:


#row 1 subplot
plt.subplot(nrows, ncols, idx)
plt.plot(data,'b*')
plt.title('Raw data')
plt.xlim((-1,10))

#row 2 subplot
plt.subplot(nrows, ncols, idx+1)
err = np.std(data, axis=1)
plt.bar(x, fit_fn(x), color='r')
plt.errorbar(x, fit_fn(x), err, fmt='*g')
plt.xlim((-1,10))  #must set limits for each subplot or else it uses defaults.

plt.show()
plt.close()


# If you want a simple way to put error bars on your bar graph (without colored errors) then use the yerr option of *bar*. Save the result as 'Figure2-Lesson8.png'

# In[78]:


#row 1 subplot
plt.subplot(nrows, ncols, idx)
plt.plot(data,'b*')
plt.title('Raw data')
plt.xlim((-1,10))

#row 2 subplot
plt.subplot(nrows, ncols, idx+1)
err = np.std(data, axis=1)
plt.bar(x, fit_fn(x), color='r', yerr=err)
plt.title('Mean data')
plt.tight_layout() #use this to adjust the spacing around subplots.
plt.xlim((-1,10))  #must set limits for each subplot or else it uses defaults.
plt.savefig('Figure2-Lesson8.png')
plt.show()
plt.close()


# ** *You should know enough now to do Exercise E. * **

# ## Lesson 9 - Basic scripts
# Writing and using scripts is the main way the typical non-programmer scientist would use Python. There are differences of opinion about the meaning of script but for my purposes it is
# - a human-readable text file containing Python commands.
# - simple structure with only a small number of functions.
# <br>
#

# You will write Python scripts with a texteditor (e.g. Atom, Notepad++) or an Integrated Development Environment (IDE) such as Spyder, which is built into the Anaconda package.<br>
# The following descriptions will assume you are using a texteditor (not an IDE) as this is the most common choice.<br>
# Comment lines begin with "#" hashtag symbol.
# You will run scripts by opening the Terminal program (or alternatives like iTerm2) and typing <br>
# <b>python <i>scriptname.py</i></b><br>
# This assumes that python has been installed and can be found ("on the path") within terminal, and that in your terminal session your current working directory is the same as the script named <i>scriptname.py</i>. This keeps things simple but you may instead want to specify the fullpathname of the script. An example on MacOS where the script is on your Desktop would be:<br>
# <b>python /Users/pradau/Desktop/myscript.py</b><br>
#
# To avoid the hassle of typing python at the start each time you should put the "shebang" at the start of your script to indicate this is a Python script. e.g.<br>
# <b>#!/usr/bin/env python</b><br>
# In the terminal, change the permissions so that the script is an "executable" on your system.
# <br>
# <b>chmod a+x myscript.py</b><br>
# This permits you to run the script as follows:<br>
# <b>./myscript.py</b>

# Unlike Matlab, you will not need to use the semi-colon (;) to suppress showing the result of each line. This is the default behavior in Python. Instead you will need to use a print() statement to show the variable or result of the calculation. (NOTE that in JN these are displayed by default.)<br>
# Here is a simple script using data of a previous lession to
# - find the mean of the data in each of the 3 categories
# - put all of these into a variable mdat
# - plot the results. <br>
#
# Enter these into your text editor and save it as "myscript.py"

# In[79]:


#!/usr/bin/env python
# this script plots the mean of data for each category
import numpy as np
import matplotlib.pyplot as plt
data=np.array([4, 14, 6, 11, 3, 14, 8, 17, 17, 12, 10, 18])
cat = np.array([1, 3, 2, 1, 2, 2, 3, 1, 3, 2, 3, 1])
#must initialize the array
mdat = np.zeros(3)
print(mdat)
mdat[0] = np.mean(data[cat==1])
mdat[1] = np.mean(data[cat==2])
mdat[2] = np.mean(data[cat==3])
print(mdat)
x = range(0,3)
plt.plot(x, mdat,'r-*')
plt.title('Mean of data for each category')
plt.savefig('Figure-script.png')


# ## Lesson 10 - Flow Control
# ### If...else

# In[80]:


#Python doesn't require an 'end' for the end of a block (e.g. if). This is accomplished by indentation.
A = 10
if A > 5:
    B = 1
else:
    B = 0
print(B)


# In[81]:


#some error checking we could have used in the earlier lesson
if len(cat) != len(data):
    print('ERROR: Data and categories are not the same length')
    sys.exit(1)  #This means exit with a return code indicating a problem.
else:
    print('No problem with array lengths.')


# ### For

# In[82]:


#This is a "non-pythonic" way to achieve the desired result. The loop is relatively slow.
# Setting the data type to integer is not usually necessary but with numpy the default is float.
#initialize an empty array of 0 elements
A = np.empty(0,dtype=int)
for i in range(1,5):
    A = np.append(A, [i*2])
    print(A)


# In[83]:


#A pythonic way to do this is to use a list comprehension to create the sequence that initializes the array.
A = np.array([i*2 for i in range(1,5)])
print(A)


# In[84]:


#Similarly we can create for loop to calculate the entries of mdat
#some data that corresponds to each cat. e.g. age
data = np.array([4,14,6,11,3,14,8,17,17,12,10,18])
#type of cat (of 3 types)
cat = np.array([1,3,2,1,2,2,3,1,3,2,3,1])
print(data[cat==1])
for i in range(0,3):
    # note the +1 because our indexing is base 0 but the categories are 1,2,3 (no 0)
    mdat[i] = np.mean(data[cat==i+1])
print(mdat)


# In[85]:


#Here is the pythonic (faster) alternative.
mdat = np.array([np.mean(data[cat==i+1]) for i in range(0,3)])
print(mdat)


# ### Switch
# There is no direct parallel to switch. This simple example could be accomplished with a Python dictionary but in the real case where there are several lines of code for each case, the typical solution would be "if...elif...else"
#

# In[86]:


A=3
if A == 1:
    print('A is one')
elif A == 3:
    print('A is three')
elif A == 5:
    print('A is five')
else:
    print('A is not one or three or five')


# ### While

# In[87]:


x = 1
y = -5
while x==1:
    y +=1
    print(y)
    if y > 1:
        x = 2



# ### Try ... Except
# The python version of Try...catch in Matlab is Try...Except. This has the identical purpose of catching errors that would otherwise cause the system to throw an error and exit.<br>
# Note that it is better to use <b>with</b> if you are working with files to provide a context manager that handles errors.

# In[88]:


A = np.array(range(1,11))
B = np.array(range(1,6))
C = np.empty(0)
print(A)
print(B)
#remember the base 0 indexing.
for i in range(0,len(A)):
    try:
        C = np.append(C,A[i] + B[i])
        print(C)
    except:
        print('B is too small')
        #sys.exit(1)  #We would use exit(1) to end the script with an error
        break  #use break to just exit the loop without quitting or returning an error code


# ## Lesson 11 - Functions
# The syntax for Python functions is slightly different than Matlab. In this example the function is <b>nearest()</b>. In the body of the script it would be called like this:<br>
# <b>index = nearest(vector, point)</b>
#
#

# In[89]:


def nearest(vector, point):
    ''' this function finds the index of the number in the vector which is closest in absolute terms to the
    point. If there is more than one match, only the 1st is returned. (This is the docstring)'''
    df = np.abs(vector - point)
    print('df',df)
    ind = np.argmin(df) #argmin() finds the index where df is minimum. Only first occurrence returned.
    return ind

vector = np.array(range(-5,5))
point = np.array(range(20,10,-1))*1.5
print(vector)
print(point)
print('nearest index', nearest(vector,point))


# Here's an example of a function **location()** with an optional argument (verbose which is by default false to minimize output to the screen). Notice that when it is not specified when the function is called, it will evaluate to False inside the function. <br> The function returns an integer (moves) and a list (new_loc) that together form a new list.

# In[90]:


def location(moves, prev_loc, move_vec, verbose=False):
    new_loc = prev_loc + move_vec
    if verbose:
        print("new location", new_loc)
    moves = moves + 1
    return moves, new_loc


move_counter = 0
knight_loc = np.array([2,20])
bishop_loc = np.array([5,6])
print("knight's move")
move_counter, knight_loc = location(move_counter, knight_loc, [1,2], True)
print("bishop's move")
move_counter, bishop_loc = location(move_counter, bishop_loc, [-1,3])
print("Number of moves", move_counter)
print('Done moving')


# More optional arguments can be used. They should then be called by name (e.g. *var*=...) and can be rearranged. e.g.

# In[91]:


def location_check(moves, prev_loc, move_vec, verbose=False, game_limit=10):
    if moves >= game_limit:
        if verbose:
            print("Game over!")
        return moves, prev_loc
    else:
        new_loc = prev_loc + move_vec
        moves = moves + 1
        if verbose:
            print("new location", new_loc)
        return moves, new_loc


move_counter = 5
knight_loc = np.array([2,20])
bishop_loc = np.array([5,6])
print("bishop's move")
move_counter, bishop_loc = location_check(move_counter, bishop_loc, [-1,3], verbose=True, game_limit = 25)
print("knight's move")
move_counter, knight_loc = location_check(move_counter, knight_loc, [1,2], verbose=True)
print("bishop's move")
# optional arguments are in different order than in function definiton. And the results are stored in a mylist instead of individually.
mylist = location_check(move_counter, bishop_loc, [-5,4], game_limit = 6, verbose=True)
print(mylist)
print("Number of moves", mylist[0])  #extract moves from list returned by function location_check()
print('Done moving')


# ### Paths
# When your program needs to check paths you will want to use the module <b>os</b>.

# In[92]:


import os
#print the current working directory
print(os.getcwd())


# In[93]:


#to see the environment variable PATH where terminal will look for files.
print(os.getenv("PATH"))


# ** *You should know enough now to do Exercise F. * **

# ## Lesson 12 - More about variables
# ### Saving and loading your data
# The easiest way to load and save (i.e. read and write in typical terminology) data is to use the Pandas module. This will take care of most of the details that you would need to do yourself if you use the lower level functions in Python (e.g. read() and write()).
#

# In[94]:


import pandas as pd
import numpy as np
x = 2
y = 3.5
z = 'Hello'
data = np.array([5,2,3])
df = pd.DataFrame()
# append each data item as a new row
#ignore_index ensures the index remains sequential after each append.
df = df.append({'Data':x}, ignore_index=True)
df = df.append({'Data':y}, ignore_index=True)
df = df.append({'Data':z}, ignore_index=True)
df = df.append({'Data':data}, ignore_index=True)
print(df)
# Save it to file called "some_stuff.csv" in comma delimited (csv) format.
#  index field is set to False so that we don't have row indices saved in the file.
df.to_csv('some_stuff.csv', index=False)


# In[95]:


#to read this data back into a new dataframe
dfnew = pd.read_csv('some_stuff.csv')
print(dfnew)


# In[96]:


#there are many options for reading/writing DataFrames
# e.g. writing a tab delimited file (sep) with no index in the first column and no column title
dfnew.to_csv('mydata.txt', sep='\t', index=False, header=False)


# The method shown above where we appended each kind of data in a single column is generally less desirable than creating a 2D dataframe where each column is a type, and each row has the values for these types. Here's an example of creating two rows in a dataframe with multiple types.
#

# In[97]:


x2 = 25
y2 = -25.7
z2 = 'World'
data2 = np.array([13,15,17])
#note that we have to make the numpy array into a list with the [] so it is the initialization knows it is
#  to go into a single cell.
row1 = {'x':x, 'y':y, 'z':z, 'data':[data]}
row2 = {'x':x2, 'y':y2, 'z':z2, 'data':data2}

df3 = pd.DataFrame(row1)
df3 = df3.append(row2, ignore_index=True)
print(df3)


# The difficulty with inserting arrays into DataFrame cells is that they are converted into strings which will require parsing for use in calculations. One would deal this in either of two ways. The first is to create a separate Dataframe with a column (or row) for each array. Then each cell would have a integer or float type and the entire column could be converted from a Series to numpy array.<br>
# This is fine if the number of elements is fixed for all the columns. But for data with arrays that may vary in length or generally for more flexibility, the alternative method is to use a list of dictionaries. The length of the list, and the size and shape of each dictionary can vary, therefore this method is useful for less structured data. Here's an example, with a 3rd record (set of data) added with a different number of elements to illustrate the benefit of this method.

# In[98]:


x3 = 26.9
y3 = 55.7
z3 = 'Third record.'
zz3 = ['an', 'example','list']
data3 = np.array([13,15,17,25.7,34.6,88])
datalist = []
#we convert the numpy arrays to float type (instead of integer) and lists to make it easier to write the file (i.e. enable JSON encoding).
data_enc = list(np.array(data,dtype=float))
data2_enc = list(np.array(data2,dtype=float))
data3_enc = list(data3)
datalist.append({'x':x, 'y':y, 'z':z, 'data':data_enc})
datalist.append({'x':x2, 'y':y2, 'z':z2, 'data':data2_enc})
datalist.append({'x':x3, 'y':y3, 'z':z3, 'zz':zz3, 'data':data3_enc})
print(datalist)


# I would recommend writing/reading this data as a [JSON](https://www.w3schools.com/python/python_json.asp) format file as it is human-readable, flexible and has well-developed supporting Python libraries.  In debugging it is useful to test that your data is correctly formatted which you can do with an online [JSON Validator](https://jsonlint.com/).<br>
# We can easily write out the data using the <b>json</b> library.

# In[99]:


import json
with open('datalist.txt', 'w') as file:
    #sort_keys, indent and separators fields make the output file easier to read.
    json.dump(datalist, file, sort_keys=True, indent=4, separators=(',', ':'))


# And then it is simple to read the data back into a new variable (read_datalist) given knowledge of the structure that was saved. Notice that using the arrays is now straight-forward: convert the stored list to a numpy array.

# In[100]:


with open('datalist.txt', 'r') as readfile:
    read_datalist = json.load(readfile)

#some examples of how to use the JSON data that was read from file.
print('read_datalist')
print(read_datalist)
print('zz variable of third record')
print(read_datalist[2]['zz'])
print('numpy array from first record')
print(np.array(read_datalist[0]['data']))
print('standard deviation from array of 2nd record')
print(np.std(read_datalist[1]['data']))


# ### Strings and cells
# There are both simple and complex ways to manipulate strings.

# In[101]:


firstname = 'Joe'
surname = 'Bloggs'
fullname = firstname + ' ' + surname
print(fullname)


# In[102]:


#here is a simple concatenation with a conversion of the age to a string
age = 25
agestring = fullname + ' ' + str(age) + ' years old'
print(agestring)


# In[103]:


#for more complex formatting it is recommended to use format()
agestring2 = fullname + ' {0} months old'.format(str(age*12))
print(agestring2)


# In[104]:


#to create a non-numeric array (eg. all strings) use the DataFrame
# this is the case where the columns of the array have no titles
stimuli = pd.DataFrame([['dog','cat','horse','rat'],['car','train','hammer','van']])
print(stimuli)


# In[105]:


#show one cell at row1 column 3. (0,2 in base 0 indexing)
print(stimuli.iloc[0][2])


# In[106]:


#can select a single row (index0)
animals = stimuli.iloc[0,:]
print(animals)


# In[107]:


#this syntax is used to get the sub-array where the value is 'cat'
print(animals[animals=='cat'])


# Here is an example of extracting a series of numbers from a DataFrame and converting it to a numpy array. This is often necessary to enable the full range of numeric functions available in numpy.
#

# In[108]:


#extract a column from dataframe df3
nums = df3['y']
print(nums)
print(type(nums))
#convert to a numpy array
realnums = np.array(nums)
print('After conversion')
print(type(realnums))
print(realnums)


# ### Structures
# The closest parallel to the Matlab 'structure' is created by using the Python 'dictionary'. We can't use pandas DataFrame because the number of elements in each cell (field) is different between records, and generally there are problems with arrays in a single DataFrame cell.

# In[109]:


word_data=['the','words','we','need']
pic_data=[1,1,5,8,9,10,25]
#here we initialize the 'data' dictionary with the first record
data_dic = {'word':word_data, 'pic':pic_data, 'subjectname':'Joe Bloggs', 'subjectage':25 }
print(data_dic)


# In[110]:


# append the second record to the first
word_data2 = ['lots','more','stuff']
pic_data2 = pic_data
data_dic2 = {'word':word_data2, 'pic':pic_data2, 'subjectname':'Jane Bloggs', 'subjectage':18 }
print(data_dic2)


# And then we create this into an analogue to 'structure' by forming a list of such records.

# In[111]:


#list of dictionaries
data = [data_dic, data_dic2]
print(data)


# In[112]:


#we can access the 2nd record's age like this
print(data[1]['subjectage'])


# ### System commands
# If you want to use these then find them in modules <b>os</b>, <b>shutil</b> or similar. As a last resort you can call them by using <b>subprocess</b>. Here are the functions I use to execute system commands under MacOS, when I can't get the results I need from the module functions.

# In[113]:


import subprocess
import shlex

def systemcall ( cmdstr ):
    ''' System call to execute command string in a shell. '''
    try:
        retcode = subprocess.call( cmdstr, shell=True)
        if retcode != 0:
            print ("Error code:", retcode)
        return retcode
    except OSError as e:
        print ("Execution failed:", e )

def systemcall_pipe( cmdstr, allow=None, disp=True ):
    ''' System call to execute command string, to get stderr and stdout output in variable proc. '''
    # this function is superior to systemcall for use with Spyder where otherwise stdout/stderr are not visible.
    # it is also needed if your main program needs to capture this output instead of only print it to terminal.
    args = shlex.split(cmdstr)
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #stdout and stderr from your process
        out, err = proc.communicate()
        retcode = proc.returncode
        if err:
            #decode the standard errors to readable form
            str_err = err.decode("utf-8")
            #Exclude error messages in allow list which are expected.
            bShow = True
            if allow:
                for allowstr in allow:
                    if allowstr in str_err:
                        bShow = False
            if bShow:
                print ("System command '{0}' produced stderr message:\n{1}".format(cmdstr, str_err))

        if disp:
            str_out = out.decode("utf-8")
            if str_out:
                print ("System command '{0}' produced stdout message:\n{1}".format(cmdstr, str_out))

        return retcode, out
    except OSError as e:
        print ("Execution failed:", e )


# In[114]:


#An example of how to use these functions to provide a directory listing.
cmdstr = "ls"
print("The systemcall method doesn't show stdout when used in Jupyter notebook but does from a script.")
retcode = systemcall(cmdstr)
print("")
print("With systemcall_pipe you can see the stdout from Jupyter notebook, and can use the results in variables:")
stdout, stderr = systemcall_pipe(cmdstr)


# ** *You should know enough now to do Exercises G and H. * **

# ## Lesson 13 - Advanced Graphs
#
# There are a vast number of graphing options which are detailed at the matplotlib.org website. I will start with a modified example from this [page](http:/matplotlib.org), then modify it to show some of the frequently used features. It is possible, but rarely needed, to get a 'handle' to a figure, axis or line. Instead just find the appropriate argument for <b>plot</b> or another function (e.g.<b>xlabel</b>) to modify the plot.
#

# In[115]:


#Simple demo with multiple subplots.
import numpy as np
import matplotlib.pyplot as plt

# x data for plots 1 and 2
x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)
# y data for plots 1 and 2
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

#arrangement of subplots
nrows = 2
ncols = 1
idx = 1
plt.subplot(nrows, ncols, idx)
#the marker/line is specified by the 'o-'
plt.plot(x1, y1, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(nrows, ncols, idx+1)
plt.plot(x2, y2, '.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')
# For saving do this
plt.savefig('subplots.png')
plt.show()
plt.close() #do this at end of each plot


# In[116]:


#revise the above graph to use a different line width,color and no symbols
plt.subplot(nrows, ncols, idx)
plt.plot(x1, y1, '-', color='cyan') #changes
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(nrows, ncols, idx+1)
plt.plot(x2, y2, '--', linewidth=4) #changes
plt.xlabel('time (s)')
plt.ylabel('Undamped')
plt.show()
plt.close() #do this at end of each plot


# ### Images
# An example of plotting a matrix as an image.

# In[117]:


# data. Each is a 1D vector with 100 elements.
x = np.linspace(0.0, 4*np.pi,100)
y = np.linspace(0.0, 2*np.pi,100)
m=100
n=1
# print(x)
#repeat x by m times in n columns. Here tile inputs 1D vector 'x' to generate a 2D matrix and repeats that row 100 times in 1 column.
# This tiling is done only to create an image from a row vector but we do need a 2D matrix for imshow()
# therefore result is a 100x100 matrix
X = np.tile(x,(m,n))
# print(X)
Y = np.tile(y,(m,n))
# generate the image 'c' from two 2D matrices (X and Y). The values are 1D (one channel) and converted to colors by a 1D color scale. (Not RGB)
c = np.sin(X) + np.cos(Y)
# extent is needed in order to get correct x,y values for the axes (instead of the matrix indices).
plt.imshow(c, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)), aspect = 'auto')
# colorbar is the color legend.
plt.colorbar()
plt.savefig('image.png')


# ## 3D Graphs
# For 3D graphs (plots) we can use the helper functions from module mplot3d. Here are some examples from the Matplotlib site.  <br>
# Here's an example 3D scatterplot.

# In[118]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
#this method requires Axes3D.The first argument determines arrangement if there are multiple subplots.

ax = fig.add_subplot(1, 1, 1, projection='3d')
# Use this instead to put red markers in separate subplot on left.
# ax = fig.add_subplot(1, 2, 1, projection='3d')

#number of points to plot for each marker type (100 red and 100 green)
n = 1000

# Set the random x coordinates to be in [-50,50], y in [0, 100].
xlow = -50
xhigh = 50
ylow = 0
yhigh = 100

#set red circle marker and zlow and zhigh.
c, m, zlow, zhigh = ('r', 'o', -50, -25)
#set the x,y coordinates randomly
xs = randrange(n, xlow, xhigh)
ys = randrange(n, ylow, yhigh)
# random values for z coordinate
#zs = randrange(n, zlow, zhigh)
# some function we wish to plot for red circles f(x,y)
zs = np.sqrt(100.0*(np.sqrt(np.absolute(xs * ys))))
# plot the 3D scatter plot of these points.
ax.scatter(xs, ys, zs, c=c, marker=m)

#use this line only if creating separate subplots, and will put green markers in right subplot.
# ax = fig.add_subplot(1, 2, 2, projection='3d')

#set green X marker and zlow and zhigh.
c, m, zlow, zhigh = ('g', 'x', -30, -5)
xs = randrange(n, xlow, xhigh)
ys = randrange(n, ylow, yhigh)
# random values for z coordinate
#zs = randrange(n, zlow, zhigh)
# some function we wish to plot for green markers f(x,y)
zs = np.sqrt(xs**2 + ys**2)
#zs = 5*np.cos(np.sqrt(xs**2 + ys**2))
ax.scatter(xs, ys, zs, c=c, marker=m)

plt.tight_layout()

plt.savefig('Figure-3Dscatter.png')
plt.show()


# Here's an example surface plot.

# In[119]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
# Create the grid of X, Y coordinates that defines the plotting area and coarseness of the plotting grid.
x = np.arange(-6, 6.25, 0.25) #x in [-6,6] with 0.25 step size.
y = np.arange(-5, 5.25, 0.25) #y in [-5,5] with 0.25 step size.
print(x)
#use the 1D x and y vectors to produce two 2D arrays. x is repeated to fill each row of X; y is repeated to fill each column of Y.
#  The 2D X is the grid of x coordinates; the 2D Y is the grid of y coordinates.
X, Y = np.meshgrid(x, y)

# Create the function you want to plot in the region defined by X,Y
R = np.sqrt(X**2 + Y**2) #radial distance from origin
# The 2D Z is the grid of z (function) values, in this case a 2D cosine wave.
Z = np.cos(R)

# Plot the surface. Different color palettes from the color map (cmap).
surf = ax.plot_surface(X, Y, Z, cmap=cm.winter, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01) #range of z axis
ax.zaxis.set_major_locator(LinearLocator(5)) #set number of z axis labels
ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f')) #string format for z labels

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('Figure-3Dsurface.png')
plt.show()


# ## Lesson 14 - Reading a Data file using Open and Read
# The best file reading method is data-dependent. My rule-of-thumb is that when the data are in a structured, matrix form (fixed number of rows and columns with a single type in each column) they are best read with Pandas DataFrame <b>from_csv</b>, and that for all other structures you will likely need to use the <b>open</b> and <b>read</b> functions. Given that Lesson 12 already covered some Pandas basics, I'll focus on the more low level functions.<br>
# Many resources on the web will recommend doing this like the following simple example, which assumes you have "myfile.txt" text file in the same directory as this python code.
#

# In[120]:


#read myfile without a context manager.
idx = 0
infile = open('myfile.txt', 'r')
for line in infile:
    #print each line of the file regardless of what's in it with prefix (linenumber:)
    # If the text file has line returns at the end of each line, which inserts empty lines in the print() output.
    print('{0}:{1}'.format(idx,line))
    idx +=1
infile.close()


# But to properly handle your file resources you should actually use a <i>context manager</i> and the keyword for doing that is <b>with</b> to ensure that files get closed properly. Some recommend using <b>close</b> to do this but unfortunately there are many cases where this statement would be missed due to errors.

# In[121]:


#read myfile by getting file handle 'infile' with a context manager.
idx = 0
with open('myfile.txt', 'r') as infile:
    for line in infile:
        #print each line of the file regardless of what's in it with prefix (linenumber:)
        # If the text file has line returns at the end of each line, empty lines will be inserted in the print() output.
        print('{0}:{1}'.format(idx,line))
        idx +=1



# In addition we would like to handle common error cases in a user-friendly fashion. Here are some typical cases. Test it out by inserting incorrect directory or filenames below.

# In[122]:


import os
import sys
mydir = "/Users/pradau/mydata"
filename = "myfile.txt"
pathname = os.path.join(mydir, filename)
if not os.path.isdir(mydir):
    print("Your directory doesn't exist:", mydir)
    sys.exit(1)
if not os.path.isfile(pathname):
    print("Your file doesn't exist at this path:", pathname)
    sys.exit(1)

with open(pathname, 'r') as infile:
    for line in infile:
        #print each line of the file regardless of what's in it with prefix (linenumber:)
        # If the text file has line returns at the end of each line, empty lines will be inserted in the print() output.
        print('{}'.format(line))



# To do something useful with the data you will typically need to parse each line. Here's an example where the lines are each put in separate lists by splitting at the whitespace (and throwing the whitespace away, such as spaces and tabs).
#

# In[ ]:


mylist = []
with open(pathname, 'r') as infile:
    for line in infile:
        mylist.append(line.split())
print(mylist)


# Either during or after the file reading you might want to do some data cleaning. Here's a simple example of cleaning during the file read that eliminates list items that are not alphanumeric.

# In[ ]:


mylist = []
with open(pathname, 'r') as infile:
    for line in infile:
        rowlist = line.split()
        #this applies a filter to the list to eliminate non-alphanumeric items.
        rowclean = [x for x in rowlist if x.isalnum()]
        mylist.append(rowclean)
print(mylist)


# Here's a basic script to read a file with either TIME or STIM or KEY in the first 4 characters, followed by data. Notice that the stimulus is an array so it can't be easily read by Pandas read_csv(). In addition we can use this method to do some organization of disorderly data. In this case the assumption is that the first instance of the TIME,STIM,KEY lines should be put in a single record and so on to group all of the data into records.

# In[ ]:


#!/usr/bin/env python
#filename = input('Enter filename: ') #User entered name
filename = 'mystim.txt'  #hard-coded name
idx = 0
trial_time = []
stimulus = []
key = []
with open(filename, 'r') as infile:
    for line in infile:
        first = line[:4]
        last = line[6:]
        if 'TIME' in first:
            trial_time.append(float(last[:5]))
        elif 'STIM' in first:
            stimulus.append(last)
        elif 'KEY ' in first:
            key.append(float(last[0]))
        else:
            #do something here if you want to handle unexpected lines
            pass
        idx +=1

for idx in range(len(key)):
    print("Record",idx)
    print("trial_time {0}  stimulus {1}  key {2}\n".format(trial_time[idx], stimulus[idx], key[idx]))


# ## Epilogue
# Those people who are still on the fence about whether it is worth their time to transition from Matlab to Python should read this [blog post](http://www.pyzo.org/python_vs_matlab.html). There's a more balanced discussion on [Quora](https://www.quora.com/How-do-MATLAB-and-Python-especially-SciPy-compare-for-scientific-computing). There's a wide world outside of Academia and the majority of people in the Data Science community are choosing Python or other languages like R. Here's a Google Trends plot showing how Python is faring in popularity. Not bad. :)

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
img=mpimg.imread('GoogleTrends.png')
plt.figure(figsize = (50,50))
plt.imshow(img)


# In[ ]:
