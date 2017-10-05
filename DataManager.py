import numpy as np
import random

class DataManager:

    def __init__(self,input_size=784,number_of_classes=10,train_filename='mnist_train.csv',test_filename='mnist_test.csv'):

        self.input_size=input_size
        self.number_of_classes=number_of_classes

        self.train_inputs=[]
        self.train_labels=[]

        self.test_inputs = []
        self.test_labels = []

        #loading MNIST data
        with open(train_filename,'r') as file:
            #for each MNIST image
            for line in file:
                #output target
                z = np.zeros(shape=(self.number_of_classes, 1), dtype=np.float32)
                l=line.split(',')
                #input normalization
                self.train_inputs.append([float(e)/255 for e in l[1:]])
                #target class
                z[int(l[0])] = 1.0
                self.train_labels.append(z)

        #same procedure for test data
        with open(test_filename,'r') as file:

            for line in file:
                z = np.zeros(shape=(self.number_of_classes, 1), dtype=np.float32)
                l=line.split(',')
                self.test_inputs.append([float(e)/255 for e in l[1:]])
                z[int(l[0])] = 1.0
                self.test_labels.append(z)


        self.cursor=0
        self.max=len(self.train_labels)-1
        self.indexes=range(0,self.max+1)
        random.shuffle(self.indexes)

        self.tcursor = 0
        self.tmax = len(self.test_labels) - 1
        self.tindexes = range(0, self.tmax + 1)
        random.shuffle(self.tindexes)

    def get_train_batch(self,size):

        if(self.cursor+size>self.max):
            random.shuffle(self.indexes)
            self.cursor=0

        batch_inputs=np.reshape([self.train_inputs[i] for i in self.indexes[self.cursor:self.cursor+size]],[size,self.input_size])
        batch_labels=np.reshape([self.train_labels[i] for i in self.indexes[self.cursor:self.cursor+size]],[size,self.number_of_classes])
        self.cursor += size

        return batch_inputs,batch_labels

    def get_test_batch(self,size):

        if(self.tcursor+size>self.tmax):
            random.shuffle(self.tindexes)
            self.tcursor=0

        batch_inputs=np.reshape([self.test_inputs[i] for i in self.tindexes[self.tcursor:self.tcursor+size]],[size,self.input_size])
        batch_labels=np.reshape([self.test_labels[i] for i in self.tindexes[self.tcursor:self.tcursor+size]],[size,self.number_of_classes])
        self.tcursor += size

        return batch_inputs,batch_labels