'''NOT A WORKING CODE'''
import torch.nn as nn  # access to nn layers
import torch.nn.functional as F  # access to activation functions (relu, sigmoid ...)
import torch.optim as optim  # access to the optimizers
import torch as T  # base torch package

'''Simple linear classifier - starter code'''
class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()

        '''three linear layers - nn.Linear(n, m): it is connecting n neurons to m neurons'''
        self.fc1 = nn.Linear(*input_dims, 128)  # the star allows to unpack tuples or list so that we can pass an arbitrary number of input dims
        self.fc2 = nn.Linear(128, 256)  # second fully connected layer
        self.fc3 = nn.Linear(256, n_classes)

        '''Adam is a stochastic gradient descent with momentum.
        An adaptive learning rate algorithm.
        self.parameters() comes from nn.Module tells us what we want to optimize.
        '''
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # loss function for multiclass classification problem. We won't use it for Deep Q
        # Best use nn.MSELoss() Mean Squared Error
        self.loss = nn.CrossEntropyLoss()

        # cuda:0 means the use the first gpu
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)  # send ENTIRE network to the device (to take advantage of gpu power)
        '''Pytorch differentiates between tensors on GPU (cuda tensors) and tensors on cpu (float/long tensors)'''

    '''Pytorch handles backpropagation algorithm for us. Just define forward algo'''
    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data))  # sigmoid is the activation function
        layer2 = F.sigmoid(self.fc2(layer1))
        '''We're not activating the output of this layer. In this case the CrossEntropyLoss
        will handle the activation for us, in such a way that the sum of all the elements adds up to 1'''
        layer3 = self.fc3(layer2)

        return layer3

    '''Handle the learning loop'''
    def learn(self, data, labels):
        '''zero out the gradient for optimizer. Pytorch keeps track of gradients between learning loops. Not needed now'''
        self.optimizer.zero_grad()
        data = T.tensor(data).to(self.device)
        '''T.Tensor() vs T.tensor(). The latter preserves the type of the data, may preserve memory. The former converts automatically to float64'''
        labels = T.tensor(labels).to(self.device)

        predictions = self.forward(data)

        cost = self.loss(predictions - labels)

        '''THESE TWO ARE CRITICAL TO THE LEARNING LOOP'''
        # backpropagate cost
        cost.backward()
        self.optimizer.step()