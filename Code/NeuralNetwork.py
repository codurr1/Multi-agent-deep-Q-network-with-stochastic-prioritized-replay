import os
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, Concatenate, Activation
from keras.optimizers import adam_v2
import tensorflow as tf
from keras import backend as K


class NeuralNetwork(object):
    
    def __init__(self, state_size, action_size, args):
        self.state_size = state_size  
        self.action_size = action_size
        self.batch_size = args['batch_size']
        self.learning_rate = args['learning_rate']
        self.num_nodes = args['number_nodes']    # number of nodes in each layer of NN
        self.model = self.build_model()
        self.model_ = self.build_model()
        self.input_shape = 3*args['uav_number'] + 6
        self.output_shape = args['uav_number'] + 2
    
    def build_model(self):
        
        # x is the input to the network 
        x = Input(shape=(3*args['uav_number'] + 6,))

        # a series of fully connected layer for estimating Q(s,a) (value of actions from that state)

        y1 = Dense(self.num_nodes, activation='relu')(x)
        y2 = Dense(self.num_nodes, activation='relu')(y1)
        z = Dense(args['uav_number'] + 2, activation="softmax")(y2)

        model = Model(inputs=x, outputs=z)
        optimizer = adam_v2.Adam(learning_rate=self.learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model
        
    
    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):  #x is the input to the network and y is the output
        self.model.fit(x, y, batch_size=len(x), sample_weight=sample_weight, epochs=epochs, verbose=verbose)
        
        
     
    def predict(self, state, target=False):
        
        if target:  # get prediction from target network
            return self.model_.predict(state)
        else: 
            return self.model.predict(state)
        
    
    def predict_one_sample(self, state, target=False):   # used for exploitation
       
        self.predict(state, target=target)
        
        
    def update_target_model(self):
        self.model_.set_weights(self.model.get_weights())
           
        
        
        
