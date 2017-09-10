import numpy as np
input_data=np.array([[1,0,0],[0,1,1],[1,0,1],[1,1,1]])
output_labels=np.array([[1,0,1,1]]).T
def activate(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
magic_matrix = 2*np.random.rand(3,1)-1#Why is it multiplied by 2 and subtract by 1.We just want random weights
for i in range(10000):
    #get guess
    guess=activate(np.dot(input_data,magic_matrix))
    #get error
    error=output_labels-guess
    #Update the weights using gradient descent
    magic_matrix+=np.dot(input_data.T,error*activate(guess,True))
print(activate(np.dot(np.array([[0,0,1]]),magic_matrix)))