import numpy as np
from math import exp
from math import log
class Network:

    def __init__(self, input_size, hidden_sizes, output_size,initialization='he_uniform',optimizer='Momentum'):

        self.input_size=input_size
        self.output_size=output_size

        #type of the optimizer
        self.optimizer=optimizer

        self.train_loss=[]
        self.train_accuracy=[]

        self.test_loss = []
        self.test_accuracy = []

        self.v_w=[]
        self.v_b=[]

        self.layers=[]
        self.biases=[]

        #Initialization of weights

        #standard initialization
        if(initialization=='standard'):
            layer = np.float32(np.random.normal(0,0.1,(hidden_sizes[0], input_size)))
        elif(initialization=='he_normal'):
            #Xavier initialization with normal dist
            layer = np.float32(np.random.normal(0, np.sqrt(1.0/float(input_size)), (hidden_sizes[0], input_size)))
        else:
            #Xavier for unifrom
            layer = np.float32(np.random.uniform(-np.sqrt(6.0 / float(input_size)),np.sqrt(6.0 / float(input_size)), (hidden_sizes[0], input_size)))

        self.layers.append(layer)

        bias = np.float32(0.0*(2*np.random.rand(hidden_sizes[0],1)-1))
        self.biases.append(bias)

        #initial velocities for momentum update
        self.v_w.append(np.float32(0*np.random.rand(hidden_sizes[0], input_size)))
        self.v_b.append(np.float32(0*np.random.rand(hidden_sizes[0], 1)))

        for i in xrange(1,len(hidden_sizes)):
            #Standard
            if (initialization == 'standard'):
                layer = np.float32(np.random.normal(0,0.1,(hidden_sizes[i], hidden_sizes[i - 1])))
            elif (initialization == 'he_normal'):
                #Xavier for normal
                layer = np.float32(np.random.normal(0, np.sqrt(1.0/float(hidden_sizes[i - 1])), (hidden_sizes[i], hidden_sizes[i - 1])))
            else:
                layer = np.float32(np.random.uniform(-np.sqrt(6.0 / float(hidden_sizes[i - 1])), np.sqrt(6.0 / float(hidden_sizes[i - 1])),
                                                 (hidden_sizes[i], hidden_sizes[i - 1])))

            self.layers.append(layer)

            bias = np.float32(0.0*(2*np.random.rand(hidden_sizes[i], 1)-1))
            self.biases.append(bias)

            #momentum velocities
            self.v_w.append(np.float32(0*np.random.rand(hidden_sizes[i], hidden_sizes[i - 1])))
            self.v_b.append(np.float32(0.0*np.random.rand(hidden_sizes[i], 1)))


        #Standard
        if (initialization == 'standard'):
            layer = np.float32(np.random.normal(0,0.1,(output_size, hidden_sizes[-1])))
        elif (initialization == 'he_normal'):
            #Xavier
            layer = np.float32(np.random.normal(0, np.sqrt(1.0/float(hidden_sizes[-1])), (output_size, hidden_sizes[-1])))
        else:
            layer = np.float32(
                np.random.uniform(-np.sqrt(6.0 / float(hidden_sizes[-1])), np.sqrt(6.0 / float(hidden_sizes[-1])),
                                  (output_size, hidden_sizes[-1])))
        self.layers.append(layer)

        bias = np.float32(0.0*(2*np.random.rand(output_size, 1)-1))
        self.biases.append(bias)

        self.v_w.append(np.float32(0 * np.random.rand(output_size, hidden_sizes[-1])))
        self.v_b.append(np.float32(0 * np.random.rand(output_size, 1)))


    def forward_pass(self,input):
        #reshape input
        out = np.float32(np.reshape(input, [self.input_size, 1]))

        #calculate outputs layer-by-layer
        i=0
        for W in self.layers[:-1]:

            out=np.add(np.matmul(W,out),self.biases[i])
            i+=1

            out=np.float32(self.relu(out))
        #softmax the result
        out=self.softmax(np.add(np.matmul(self.layers[-1],out),self.biases[-1]))

        return out


    def calculate_test_loss_acc(self,batch,batch_target,batch_size):

        total_error = 0
        total_accuracy = 0

        for batch_element in xrange(0,batch_size):
            input = batch[batch_element]
            target = batch_target[batch_element]

            x_l=self.forward_pass(input)
            t = np.reshape(target, [self.output_size, 1])

            total_error += self.calculate_loss(x_l, t)
            total_accuracy += self.calculate_accuracy(x_l, t)

        self.test_loss.append(total_error / batch_size)
        self.test_accuracy.append(float(total_accuracy) / batch_size)

    #Basically the same forward pass, but stores all the layer outputs (to be used in training)
    def forward_pass_with_layers(self,input):

        out=np.float32(np.reshape(input,[self.input_size,1]))

        layer_outputs=[]

        i=0
        for W in self.layers[:-1]:
            out=np.add(np.matmul(W,out),self.biases[i])
            i+=1

            out=np.float32(self.relu(out))

            layer_outputs.append(out)

        out = self.softmax(np.float32(np.add(np.matmul(self.layers[-1], out), self.biases[-1])))

        layer_outputs.append(out)

        return out,layer_outputs

    def train(self,batch,batch_target,learning_rate,batch_size):

        total_layers=[]
        total_biases=[]
        total_error=0
        total_accuracy=0

        for batch_element in xrange(0, batch_size):

            #get a single instance
            input=batch[batch_element]
            target=batch_target[batch_element]
            newlayers=[]
            newbiases=[]

            #run a forward pass
            x_l,h_outputs=self.forward_pass_with_layers(input)
            t=np.reshape(target,[self.output_size,1])

            #calculate accuracy and loss
            total_error+=self.calculate_loss(x_l,t)
            total_accuracy+=self.calculate_accuracy(x_l,t)

            #calculate weight updates for the last layer
            delta=np.subtract(x_l,t)
            upd=np.matmul(delta,np.transpose(h_outputs[-2]))
            #store this values
            newlayers.append(upd)
            newbiases.append(delta)

            #do a backpropogation
            for i in xrange(len(self.layers)-2,-1,-1):
                #calculate delta
                delta=np.multiply(np.matmul(np.transpose(self.layers[i+1]),delta),self.diff_relu(h_outputs[i]))
                #calculate and store weight & bias updates
                if(i>0):
                    upd = np.matmul(delta, np.transpose(h_outputs[i-1]))
                else:
                    upd = np.matmul(delta, np.transpose(np.reshape(input,[self.input_size,1])))

                newlayers.append(upd)

                newbiases.append(delta)

            # store add weigt& bias updates to values from previous instances
            if(len(total_biases)==0):

                for each in newlayers:
                    total_layers.append(each)
                for each in newbiases:
                    total_biases.append(each)

            else:

                total_layers=np.add(total_layers,newlayers)
                total_biases=np.add(total_biases,newbiases)

        #calculate average weight and bias updates
        total_layers=np.divide(total_layers,float(batch_size))
        total_biases=np.divide(total_biases,float(batch_size))

        #do weight updates with Momentum or Stochastic Gradient Descent
        for i in xrange(0,len(total_layers)):
            #self.layers[len(self.layers)-i-1]=newlayers[i]
            #self.biases[len(self.biases)-i-1]=newbiases[i]
            if(self.optimizer=="SGD"):
                #SGD
                self.layers[len(self.layers) - i - 1] = np.subtract(self.layers[len(self.layers) - i - 1], learning_rate * total_layers[i])
                self.biases[len(self.biases) - i - 1] = np.subtract(self.biases[len(self.biases) - i - 1], learning_rate * total_biases[i])
            else:
                #Momentum
                v=np.add(0.9 * self.v_w[len(self.layers) - i - 1],
                       learning_rate * total_layers[i])
                self.v_w[len(self.layers) - i - 1]=v
                self.layers[len(self.layers) - i - 1] = np.subtract(self.layers[len(self.layers) - i - 1],
                                                                    self.v_w[len(self.layers) - i - 1])
                self.v_b[len(self.layers) - i - 1]=np.add(0.9 * self.v_b[len(self.layers) - i - 1],
                       learning_rate * total_biases[i])
                self.biases[len(self.biases) - i - 1] = np.subtract(self.biases[len(self.biases) - i - 1],
                                                                    self.v_b[len(self.layers) - i - 1])

        #append training loss and accuracy
        self.train_loss.append(total_error/batch_size)
        self.train_accuracy.append(float(total_accuracy)/batch_size)

    def sigmoid(self,x):

        for i in xrange(0,len(x)):
            x[i][0]=1.0/(1.0+np.exp(-x[i][0]))
        return x

    def relu(self, x):
        for i in xrange(0,len(x)):
            if(x[i][0]<0):
                x[i][0]=0.0
        return x

    def diff_relu(self,x):
        y=[]
        for i in xrange(0, len(x)):
            if (x[i][0] <= 0):
                y.append([0.0])
            else:
                y.append([1.0])
        return y

    def softmax(self,v):

        omega=0

        for e in v:
            try:
                omega+=exp(e[0])
            except:
                t=np.argmax(v)
                tt=np.zeros((4,1),dtype=np.float32)
                tt[t]=[1.0]
                return tt


        y=[]

        for e in v:
            y.append([exp(e[0])/omega])

        return y

    def calculate_loss(self,output,target):
        res=0
        for i in xrange(0,len(target)):
            res-=float(target[i][0])*log(output[i][0]+0.001)
        return res

    def calculate_accuracy(self,output,target):
        res=0
        if(np.argmax(target)==np.argmax(output)):
            res+=1
        return res

    def calculate_batch_accuracy_loss(self,inputs,targets):
        tot_acc=0.0
        tot_loss=0.0
        for i in xrange(0,len(inputs)):
            t=np.reshape(targets[i],[self.output_size,1])
            out=self.forward_pass(inputs[i])
            tot_acc+=self.calculate_accuracy(out,t)
            tot_loss+=self.calculate_loss(out,t)


        return tot_acc/len(inputs),tot_loss/len(inputs)