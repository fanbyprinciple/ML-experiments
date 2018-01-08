import numpy as np

#define parameters in an understandable format
x = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0],
            [1],
            [1],
            [0]])

print(x)

print(y)

#define hyperparameters and build models

num_epochs = 60000

#initialise probable values for wieght values
syn0 = 2*np.random([3,4])-1
syn1 = 2*np.random([4,1])-1

print(syn0)
print(syn1)

#define a sigmoid that takes a nonlinear value and "squashes " it into a linear one
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np, exp(-x))

#train model
for j in range(num_epochs):
    #feed forward through layers 0,1 and 2
    l0 = x
    l1 = nonlin(np.dot(l0,syn0)) 
    l2 = nonlin(np.dot(l1,syn1)) 

    #how much did we miss the target value?
    l2_error = y - l2

    #in what direction is the target value?
    l2_delta = l2_error*nonlin(l2,deriv=True)

    #how much did each l1 value contribute to l2 error
    l1_error = l2_delta.dot(syn1.T)

    l1_delta= l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)