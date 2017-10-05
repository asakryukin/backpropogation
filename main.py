from Network import Network
from DataManager import DataManager
from numpy import random
import numpy
from matplotlib import pyplot

#Initialize Data manager class
Data=DataManager()

#batch size
batch_size=50

#Number of input neurons
number_of_inputs=784

#Number of output neurons
number_of_outputs=10

#Array, each entry indicates number of neurons in a layer
hidden_layer_sizes=[100,80]

###Network initization
N1=Network(number_of_inputs,hidden_layer_sizes,number_of_outputs,'he_uniform','Momentum')

learning_rate=0.001

#total number of images in training data
total=50000
#number of iterations in one epoch
to_cover_one=total/batch_size

n_epochs=3

for i in xrange(0,n_epochs*to_cover_one):

    #get input batch
    inputs,target=Data.get_train_batch(batch_size)

    #update weights
    N1.train(inputs,target,learning_rate,batch_size)

    #print once epoch is covered
    if(i%to_cover_one==0):
        print "Epoch "+str(i/to_cover_one)+" finished\n"
        ac,ls=N1.calculate_batch_accuracy_loss(inputs,target)
        print "Accuracy:"+str(ac)+" Loss:"+str(ls)

#Plotting training losses
pyplot.plot(N1.train_loss,label='Network 1')
pyplot.title("Training loss")
pyplot.ylabel("loss magnitude")
pyplot.xlabel("iteration")
pyplot.legend(loc='upper left')
pyplot.show()

#plotting test losses
#did not calculate here, but you can just add N1.test after every training itiration or less frequent
#pyplot.plot(N1.test_loss,label='Network 1')
#pyplot.title("Testing loss")
#pyplot.ylabel("loss magnitude")
#pyplot.xlabel("iteration")
#pyplot.legend(loc='upper left')
#pyplot.show()

#Plotting train/test accuracies for each network
pyplot.plot(N1.train_accuracy, label="Training")
pyplot.plot(N1.test_accuracy, label='Testing')
pyplot.title("Network 1 Accuracy")
pyplot.ylabel("Accuracy")
pyplot.xlabel("iteration")
pyplot.legend(loc='upper left')
pyplot.show()