# © All rights reserved. University of Edinburgh, United Kingdom
# SMILE Project, 2019

import numpy as np
import pandas as pd
import os
from pathlib import Path

# from keras.models import Model
# from keras.layers import Conv1D, Flatten, Reshape, Input, multiply, Lambda
from tflite_runtime.interpreter import Interpreter


class NILM_Model(object):

    def __init__(self, appliance, model_path='./'):

        self.appliance = appliance

        # Models for these appliances exist
        self.allowed_appliances = ['dishwasher', 'electriccooker', 'electricshower', 'kettle', 'microwave',
                                   'washingmachine']

        # Specify where the model weights are located and create a filename generator function
        self.model_path = Path(model_path)
        #self.model_weights = lambda a: self.model_path / Path('layers_7_rate_10_{}.tflite'.format(a))

        # Specify statistics which are need for predictions
        self.stats = {'mains_mean':386.63,
                      'mains_std':737.2,
                      'dishwasher_mean':888.492409282603,
                      'electriccooker_mean':1648.82006368878,
                      'electricshower_mean':9653.03708202749,
                      'kettle_mean':2645.57182476043,
                      'microwave_mean':1321.60439026507,
                      'washingmachine_mean':480.743039815068,
        }

        # Load the model and weights
        print('Loading model for {}.'.format(self.appliance))
        #self.model = self.load_model(self.appliance)
        self.interpreter = Interpreter(os.path.join(model_path, 'layers_7_rate_10_{}.tflite'.format(self.appliance)))
        #self.interpreter = Interpreter('/home/blackwood/RunLiveNILM/TFLiteModels/layers_7_rate_10_{}.tflite'.format(self.appliance))


    def init_model(self):
        initial_filter_size = 9
        print("init")

    def old_init_model(self):
        """ Initialise the model architecture. """
        # Parameter settings
        initial_filter_size = 9
        num_dilation_layers = 7
        num_filters = 128
        dilated_filter_size = 3

        # Compute the receptive field of the model
        receptive_field = initial_filter_size + 2 ** (num_dilation_layers + 2) - 4

        # Input and output windows
        input_window_length = 2 * receptive_field - 1
        output_window_length = receptive_field

        # Offset defining which part of the output is returned
        offset = (input_window_length - output_window_length) // 2

        # Define the model inputs
        main_input = Input(shape=(input_window_length,), name='main_input')
        targets_mask = Input(shape=(output_window_length,), name='targets_mask')

        # Reshape the input to match Keras' requirements
        x = Reshape((input_window_length, 1), input_shape=(input_window_length,))(main_input)

        # Apply the initial convolution
        x = Conv1D(num_filters, initial_filter_size, padding='same', activation='relu', dilation_rate=1)(x)

        # Add the dilated convolutions
        for n in range(num_dilation_layers):
            x = Conv1D(num_filters, dilated_filter_size, padding='same', activation='relu',
                       dilation_rate=2 ** (n + 1))(x)

        # Final convolution layers
        x = Conv1D(num_filters, 1, padding='same', activation='relu')(x)
        x = Conv1D(1, 1, padding='same', activation=None)(x)

        # Bring the shape back to the original input shape and selec the bits which should be returned
        x = Reshape((input_window_length,), input_shape=(input_window_length, 1))(x)
        x = Lambda(lambda x: x[:, offset:-offset], output_shape=(output_window_length,))(x)

        # Mask the parts of the output which are deemed unsuitable (as defined by the input)
        main_output = multiply([x, targets_mask])

        # Create the Model architecture
        model = Model(inputs=[main_input, targets_mask], outputs=[main_output])

        return model

    def load_model(self, appliance):
        """ Load the model for an appliance. """
        if not appliance in self.allowed_appliances:
            raise ValueError('No model for {}.'.format(appliance))

        model = self.init_model()
        model.load_weights(self.model_weights(appliance))

        return model

    def predict(self, ts):
        if not isinstance(ts, pd.Series):
            raise ValueError('The input must be a pd.Series.')
        if not (ts.index.freq == '10s'):
            raise ValueError('The input must be at 10s frequency.')
        if not len(ts) == 1033:
            raise ValueError('The input length must be 1033. Got {}'.format(len(ts)))

        #input_data = np.float32(input_data)
        # Prepare the input data
        data =  {'main_input': np.float32((ts.values.reshape(1,-1)-self.stats['mains_mean']) / self.stats['mains_std']),
                'targets_mask': np.float32(np.ones((1,517)))}
        minp = {'main_input': (ts.values.reshape(1,-1)-self.stats['mains_mean']) / self.stats['mains_std'] }
        tmask = {'targets_mask': np.ones((1,517))}
        darr = []
        darr.append(minp)
        darr.append(tmask)

        # Predict and scale
        #predictions = self.model.predict(data) * self.stats['{}_mean'.format(self.appliance)]
        signature_defs = self.interpreter.get_signature_list()
        print(signature_defs)
        sigrun = self.interpreter.get_signature_runner()
        #print(darr)
        predictions = sigrun(**data)
        #predictions = sigrun({'inputs': darr})
        #predictions = sigrun(darr)
        if 'multiply' in predictions:
           predictions = predictions['multiply'] * self.stats['{}_mean'.format(self.appliance)]
        if 'multiply_1' in predictions:
           predictions = predictions['multiply_1'] * self.stats['{}_mean'.format(self.appliance)]
        if 'multiply_2' in predictions:
           predictions = predictions['multiply_2'] * self.stats['{}_mean'.format(self.appliance)]
        if 'multiply_3' in predictions:
           predictions = predictions['multiply_3'] * self.stats['{}_mean'.format(self.appliance)]
        if 'multiply_4' in predictions:
           predictions = predictions['multiply_4'] * self.stats['{}_mean'.format(self.appliance)]

        # Predicted power should never be smaller zero
        predictions[predictions < 0] = 0
        #print("predictions: ",predictions)

        return pd.Series(predictions.reshape(-1,),
                         index=pd.date_range(start=ts.index[258], end=ts.index[-259], freq='10s'),
                         name='predictions')
