import sys
import numpy as np
import pandas as pd

#to avoid the deviation in result
np.random.seed(49)

#generating numpy array from input .csv file
def array_generation(f1,f2,f3):
    #converting input from csv to numpy array
    #The genfromtxt() is used to load data from a text file, with missing values handled by it
    train_image = np.genfromtxt(f1, delimiter= ",")
    train_label = np.genfromtxt(f2, delimiter= ",")
    test_image = np.genfromtxt(f3, delimiter= ",")
    
    #normalizing the image pixels array so that the value is between 0 and 1 (standard state)
    #In order to change the dtype of the given array object, we will use numpy. astype() function
    #images are in sizes of pixels from 0-255
    train_image = train_image.astype(float)/255.
    test_image = test_image.astype(float)/255.

    #converting the labels to int
    train_label = train_label.astype(int)

    #making it into an nx1 array
    #flattening the images
    #The reshape() function is used to give a new shape to an array without changing its data.
    train_image = train_image.reshape(train_image.shape[0],-1)
    test_image = test_image.reshape(test_image.shape[0],-1)

    return train_image,train_label,test_image

#splitting the dataset into batches
def splitbatches(train_image,train_label,batch):
    for i in range(0, len(train_image), batch):
        index = slice(i, i+batch)
        #instead of returning the output, yield returns a generator that can be iterated upon.
        yield train_image[index], train_label[index]

#class for hidden layer
class HiddenLayer():
    #Initializing weights and bias for hidden layer
    def __init__(self,input,output):
        #xavier initialization
        self.weight = np.random.randn(input,output) * np.sqrt(2/(input+output)) 
        self.bias = np.zeros(output)

    #Selecting one batch of data and calculating forward pass
    def forward(self,train_image):
        #computing the dot product of train_images with the weights + bias
        return np.dot(train_image,self.weight) + self.bias

    #to learn and optimize weights and biases using back prop
    def backward(self, train_image, train_label):
        #learning rate tweaked accordingly for better performance
        alpha = 0.15
        #we are tweaking the inputs here and adjusting them accoringly in backward prop
        gradient_input = np.dot(train_label,self.weight.T)
        #we are tweaking the weights here and adjusting them accoringly in backward prop
        gradient_weight = np.dot(train_image.T, train_label)
        #we are tweaking the biases here and adjusting them accoringly in backward prop
        gradient_bias = train_label.mean(axis=0)*train_image.shape[0]
        #achieving local minima by updating weight
        self.weight = self.weight - alpha * gradient_weight 
        #updating the bias for achieving local minima
        self.bias = self.bias - alpha * gradient_bias 
        return gradient_input

#to get an interesting function from neural network and to avoid linear function with weights and biases
#class for activation layer
#leaky Rectified linear unit
#if it passes the threshold 0 it will be same as f(a) else it will be inactive ie 0
class LeakyReLU():
    def forward(self,input):
        #ReLU(x) is x if x>0, is 0 if x<=0
        alpha = 0.01
        return np.maximum(alpha*input,input)

    def backward(self,input,output):
        #ReLU(x) is x if x>0, is 0 if x<=0
        t_input = input > 0
        return (t_input*output)

#class for the NueralNetwork for better performance
class NueralNetwork():
    def __init__(self):
        self.nn = list()
        #hidden layer 1: input is 28*28(784) -> 600 first hidden layer
        self.nn.append(HiddenLayer(784,600)) 
        #LeakyReLU activation 1
        self.nn.append(LeakyReLU()) 
        #hidden layer 2: input is 600 -> 200 second hidden layer
        self.nn.append(HiddenLayer(600,256)) 
        #LeakyReLU activation 2
        self.nn.append(LeakyReLU())
        #output:column containing probabilities of the handwriten digit to be a number from 0-9 
        self.nn.append(HiddenLayer(256,10)) 

    #softmax function for probability between 0 and 1 for digits 0-9
    #exp to that node divided by the sum of exp to all of the nodes
    #softmax for final output layer controls exactly how the weights of the computational graph are adjusted during training
    def softmax(self,input):
        exponent = np.exp(input)
        return exponent/np.sum(exponent, axis = 1, keepdims = True)  

    #gradient cross entropy function to quantify how acurate the model predictions were after feedforward step
    def gradient_cross_entropy(self,obtained_input,expected_output):
        c = obtained_input.shape[0]
        l = np.zeros((obtained_input.shape[0],obtained_input.shape[1]))
        l[range(c),expected_output] = 1 
        #probability of the obtained input
        s = self.softmax(obtained_input) 
        return (- l + s)/c

    #Forward calculation
    def forward(self,train_image):
        a = []
        #feedforward propagation for loop
        for i in self.nn:
            a.append(i.forward(train_image))
            train_image = a[-1]
        return a

    #Training the dataset
    def train(self, train_image, train_label):
        l = self.forward(train_image)
        output = l[-1]
        #Adding the given data to the activation found
        layer = [train_image]+l  
        loss_gradient = self.gradient_cross_entropy(output,train_label) 

        for index in range(len(self.nn))[::-1]:
            curr = self.nn[index]
            loss_gradient = curr.backward(layer[index],loss_gradient) 

        

    #Predicting the given image
    def predict(self,test_image):
        output = []
        input = test_image
        for i in self.nn:
            output.append(i.forward(input))
            input = output[-1]
        res = self.softmax(output[-1])
        return res.argmax(axis=-1)


if __name__ == "__main__":
    #if the len of cmd is 1
    if len(sys.argv) == 1:
        train_image, train_label, test_image = array_generation('train_image.csv','train_label.csv','test_image.csv')
    else:
        #take the filenames as command line arguments
        f1, f2, f3 = sys.argv[1:4]
        train_image,train_label,test_image = array_generation(f1,f2,f3)
    nn_obj = NueralNetwork()
    #epochs run 150 times ie 150*60000 
    #no of epochs 150 because geting more accuracy
    for i in range(150):
        #batch size 100
        for x,y in splitbatches(train_image,train_label,batch = 100):
            nn_obj.train(x,y)
    res = nn_obj.predict(test_image) 
    f = open("test_predictions.csv","w") 
    f.write("\n".join([str(x) for x in res]))
    f.close()

