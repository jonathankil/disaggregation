"""
Created on 19/06/18
@author Cillian Brewitt
"""

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

class NilmModel:
    """ abstract class for NILM models """

    model_id = NotImplemented
    supported_appliances = NotImplemented  # list of appliancetypes the model can disaggregate
    sample_rate = NotImplemented
    offset = 0  # samples required before and after prediction window

    def predict(self, aggregate_readings, appliances):
        raise NotImplementedError()

    def process_appliance_list(self, appliances):
        appliances = [a.lower() for a in appliances]

        for appliance in appliances:
            assert appliance in self.supported_appliances

        return appliances


class NilmFullyConv(NilmModel):

    sql_insert = """
    INSERT INTO model (bamtype, script, modelfile, modeltype, description) VALUES (
    'inferred',
    'trunk/software/ml/nilm_nn/nn/s2p_fully_conv.py',
    'fully_conv_[appliancetype].h5',
    'household electrical appliance model',
    'Separate fully convolutional neural network for each appliancetype.
    Data from gold homes before 5th May 2018 was used for training.'
    );"""

    model_id = 2
    supported_appliances = ['kettle', 'microwave', 'washingmachine', 'dishwasher']
    sample_rate = 8
    offset = 299

    output_window_length = 599
    input_window_length = output_window_length + 2 * offset

    def __init__(self, model_path):
        self.keras_models = {}
        for appliance in self.supported_appliances:
            self.keras_models[appliance] = self.keras_model()
            self.keras_models[appliance].load_weights(model_path + self.file_name(appliance))

        self.appliance_stats = pd.read_csv(os.path.dirname(__file__)
                                           + '/../nilm_nn/ideal/appliance_stats.csv')

    def predict(self, aggregate_readings, appliances):
        """infer appliance readings given aggregate (mains electricity) readings"""

        appliances = self.process_appliance_list(appliances)

        receptive_field = 2 * self.offset
        inputs, _, targets_mask, _, time = home_s2s_multi_appliance(
            aggregate_readings, appliances, self.sample_rate, receptive_field, self.output_window_length)[:5]

        if inputs.shape[0] > 0:

            # preprocess / normalise
            targets_mask = targets_mask.astype(np.bool).reshape((-1, self.output_window_length))
            inputs = np.nan_to_num(inputs)
            inputs = (inputs - inputs.mean(axis=1).reshape((-1, 1))) / ideal_meta.mains_std

            appliance_readings = pd.DataFrame(index=pd.to_datetime(time.flatten(), unit='s'))

            for appliance in appliances:
                # make predictions
                predictions = self.keras_models[appliance].predict([inputs, ~targets_mask])

                mean_on_power = float(self.appliance_stats.loc[
                                          self.appliance_stats.appliancetype==appliance, 'mean_on_power'])
                predictions = predictions * mean_on_power
                predictions[targets_mask] = np.nan
                predictions[predictions < 0] = 0

                appliance_readings[appliance] = predictions.flatten()
            appliance_readings = appliance_readings.dropna()
            return appliance_readings

    def keras_model(self):
        main_input = Input(shape=(self.input_window_length,), name='main_input')
        x = Reshape((self.input_window_length, 1),  input_shape=(self.input_window_length,))(main_input)

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

        x = Reshape((self.input_window_length,),  input_shape=(self.input_window_length, 1))(x)

        x = Lambda(lambda x: x[:, self.offset:-self.offset], output_shape=(self.output_window_length,))(x)

        targets_mask = Input(shape=(self.output_window_length,), name='targets_mask')
        main_output = x
        single_model = Model(inputs=[main_input, targets_mask], outputs=[main_output])

        return single_model

    def file_name(self, appliance):
        return 'fully_conv_{0}.h5'.format(appliance)

# if __name__ == '__main__':
#     # for testing only
#     from data_store import HomeReadingStore
#
#     with HomeReadingStore() as s:
#         aggregate_readings = s.get_readings(73)['mains_apparent']
#
#     model = NilmFullyConv('models/')
#
#     appliance_readings = model.predict(aggregate_readings, ['kettle'])
