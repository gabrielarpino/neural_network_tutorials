import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

printed = False

for iter in xrange(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

    if printed == False:

        print "np.dot(l0.T,l1_delta)"
        print np.dot(l0.T,l1_delta)
        print "l0.T"
        print l0.T
        print "l1_delta"
        print l1_delta
        print "syn0"
        print syn0
        print "l1_error:"
        print l1_error
        print "current l1"
        print l1
        print "l1_delta"
        print l1_delta
        printed = True



print "last l1_error:"
print l1_error
print "Output After Training:"
print l1

#now with this newly trained network, see what it does with new input
#input of [0,1,0]

l0 = np.array([0,0,0])
answer = nonlin(np.dot(l0,syn0),False)

print "new answer:", answer
