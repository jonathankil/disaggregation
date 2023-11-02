import sys
import os
sys.path.insert(0, '../nilm_nn/ideal')
sys.path.insert(0, '../data_conversion')

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Conv1D, Reshape, Input, multiply, Lambda

from generate_s2s_dataset import home_s2s_multi_appliance
import ideal_meta

from nilm_model import NilmFullyConv

class NilmFullyConvMasked(NilmFullyConv):

    sql_insert = """
    INSERT INTO model (bamtype, script, modelfile, modeltype, description, usedisplayappliances, energydisplayappliances) VALUES (
    'inferred',
    'trunk/software/ml/nilm_nn/nn/fully_conv_separate_valid.py',
    'fully_conv_masked_[appliancetype].h5',
    'household electrical appliance model',
    'Separate fully convolutional neural network for each appliancetype. Target timesteps near large gaps in sensor
    readings where masked during training. Data from gold homes up to 30th June 2018 was used for training.',
    'kettle,microwave,washingmachine,dishwasher,electricshower,electriccooker',
    'kettle,microwave,washingmachine,dishwasher,electricshower,electriccooker'
    );"""

    model_id = 5
    supported_appliances = ['kettle', 'microwave', 'washingmachine', 'dishwasher', 'electricshower', 'electriccooker']
    sample_rate = 8
    offset = 299
    output_window_length = 599
    input_window_length = output_window_length + 2 * offset

    def file_name(self, appliance):
        return 'fully_conv_masked_{0}.h5'.format(appliance)

    def keras_model(self):
        main_input = Input(shape=(self.input_window_length,), name='main_input')
        x = Reshape((self.input_window_length, 1), input_shape=(self.input_window_length,))(main_input)

        x = Conv1D(128, 9, padding='same', activation='relu', dilation_rate=1)(x)
        x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=2)(x)
        x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=4)(x)
        x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=8)(x)
        x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=16)(x)
        x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=32)(x)
        x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=64)(x)
        x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=128)(x)
        x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=256)(x)
        x = Conv1D(256, 1, padding='same', activation='relu')(x)
        x = Conv1D(1, 1, padding='same', activation=None)(x)

        x = Reshape((self.input_window_length,), input_shape=(self.input_window_length, 1))(x)

        x = Lambda(lambda x: x[:, self.offset:-self.offset], output_shape=(self.output_window_length,))(x)

        targets_mask = Input(shape=(self.output_window_length,), name='targets_mask')
        main_output = multiply([x, targets_mask])
        single_model = Model(inputs=[main_input, targets_mask], outputs=[main_output])
        return single_model