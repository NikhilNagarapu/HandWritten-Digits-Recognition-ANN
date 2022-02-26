import sys
import numpy as np

#seed is a function that sets the random seed of the NumPy pseudo-random number generator. It provides an essential input that enables NumPy to generate pseudo-random numbers for random processes.
np.random.seed(49)
#to avoid the deviation in result

#generating numpy array from input .csv file
def array_generation(f1,f2,f3):
    #converting input from csv to numpy array
    #The genfromtxt() used to load data from a text file, with missing values handled
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
class Hidden():
    #Initializing weights and bias for hidden layer
    def __init__(self,input,output):
        #xavier initialization
        #It is the weights initialization technique that tries to make the variance of the outputs of a layer to be equal to the variance of its inputs.
        self.weight = np.random.randn(input,output) * np.sqrt(2/(input+output)) 
        self.bias = np.zeros(output)

    #Selecting one batch of data and calculating forward pass
    def forward(self,train_image):
        #computing the dot product of train_images with the weights + bias
        return np.dot(train_image,self.weight) + self.bias

    #to learn and optimize inputs,weights and biases using back prop
    def backward(self, train_image, train_label):
        #learning rate tweaked accordingly
        learn_rate = 0.15
        #we are tweaking the inputs here and adjusting them accoringly in backward prop
        gradient_input = np.dot(train_label,self.weight.T)
        #we are tweaking the weights here and adjusting them accoringly in backward prop
        gradient_weight = np.dot(train_image.T, train_label)
        #we are tweaking the biases here and adjusting them accoringly in backward prop
        gradient_bias = train_label.mean(axis=0)*train_image.shape[0]
        #achieving local minima by updating weight
        self.weight = self.weight - learn_rate * gradient_weight 
        #updating the bias for achieving local minima
        self.bias = self.bias - learn_rate * gradient_bias 
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
        true_input = input > 0
        return (true_input*output)

#class for network
class Network():
    def __init__(self):
        self.network = list()
        #hidden layer 1
        self.network.append(Hidden(784,300)) 
        #LeakyReLU activation 1
        self.network.append(LeakyReLU()) 
        #hidden layer 2
        self.network.append(Hidden(300,200)) 
        #LeakyReLU activation 2
        self.network.append(LeakyReLU())
        #output:column containing probabilities of the handwriten digit to be a number from 0-9 
        self.network.append(Hidden(200,10)) 

    #softmax function for probability between 0 and 1 for digits 0-9
    #exp to that node divided by the sum of exp to all of the nodes
    #softmax for final output layer
    def softmax(self,input):
        exponent = np.exp(input)
        return exponent/np.sum(exponent, axis = 1, keepdims = True)  

    #Cross entropy Loss(Cost) function
    def cross_entropy_loss(self,obtained_output,expected_output):
        c = expected_output.shape[0]
        soft = self.softmax(obtained_output) #probability of the obtained input
        log_cal = -np.log(soft[range(c),expected_output])
        loss = np.sum(log_cal)/c #loss equation
        return loss

    def gradient_cross_entropy(self,obtained_input,expected_output):
        c = obtained_input.shape[0]
        logits = np.zeros((obtained_input.shape[0],obtained_input.shape[1]))
        logits[range(c),expected_output] = 1 #assigning value=1 for given label index
        soft = self.softmax(obtained_input) #probability of the obtained input
        return (- logits + soft)/c

    #Forward calculation
    def forward(self,train_image):
        activation = []

        for each in self.network:
            #feedforward propagation for loop
            activation.append(each.forward(train_image))
            train_image = activation[-1]
        return activation

    #Training the dataset
    def train(self, train_image, train_label):
        list_of_activation = self.forward(train_image)
        obtained_output = list_of_activation[-1]
        layer_with_activation = [train_image]+list_of_activation #Adding the given data to the activation found  
        loss_grad = self.gradient_cross_entropy(obtained_output,train_label) 

        for index in range(len(self.network))[::-1]:
            current = self.network[index]
            loss_grad = current.backward(layer_with_activation[index],loss_grad) #backward calculation

        loss = self.cross_entropy_loss(obtained_output,train_label) #loss calculation ~ not used anywhere

    #Predicting the given image
    def predict(self,test_image):
        output = []
        input = test_image
        for i in self.network:
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
    net_object = Network()
    #epochs run 20 times ie 20*60000
    for i in range(20):
        for x,y in splitbatches(train_image,train_label,batch = 50):
            net_object.train(x,y)
    result = net_object.predict(test_image) #predicted result obtained as array
    f = open("test_predictions.csv","w") #file writer
    f.write("\n".join([str(x) for x in result]))
    f.close()

