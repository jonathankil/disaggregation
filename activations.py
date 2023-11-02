"""
Created on 19/03/18
@author Cillian Brewitt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '../inferred_bam')
import numpy as np
import pandas as pd
import datetime as dt
import gc
import json
import re
from electric import submeter

class ActivationDetector:

    def __init__(self, appliance, rulesfile='../data_conversion/rules.json', sample_rate=1):
        
        self.rulesfile=rulesfile
        
        with open(rulesfile) as data_file:
            rules = json.load(data_file)['rules']

        self.rule = None
        for rule in rules:
            pattern = re.compile(rule["appliance"])
            if not (pattern.match(appliance) is None):
                self.rule = rule
                break

        assert self.rule is not None
        self.appliance = appliance

        self.min_off_duration=int(self.rule["min_off_duration"])
        self.min_on_duration=int(self.rule["min_on_duration"])
        self.max_on_duration=int(self.rule["max_on_duration"])
        self.on_power_threshold=int(self.rule["on_power_threshold"])
        
        # minumum activation energy in joules
        
        if "min_energy" in self.rule:
            self.min_energy = int(self.rule["min_energy"])
        else:
            self.min_energy = 0

        self.sample_rate = sample_rate

    def get_activations(self, readings):
        """Get start, end and energy (joules) of appliance activations"""
        #resample and add buffer to start and end
        buffer = dt.timedelta(seconds=self.min_off_duration+self.sample_rate*2)
        if (len(readings.index)==0):
            return;
        start_time = readings.index[0]
        end_time = readings.index[-1]

        readings.loc[start_time - buffer] = 0
        readings.loc[end_time + buffer] = 0
        readings.sort_index(inplace=True)

        readings = (readings
                    .resample('{0}S'.format(self.sample_rate))
                    .fillna('nearest', 1)
                    .fillna(0))

        end = readings.index[-1]
        readings.columns=['values']
        #print(readings.head(15))
        on_offs = submeter().get_ons_and_offs(readings, end, None,
                                              min_off_duration=self.min_off_duration,
                                              min_on_duration=self.min_on_duration,
                                              max_on_duration=self.max_on_duration,
                                              on_power_threshold=self.on_power_threshold)
        
        starts = on_offs.time[on_offs.state_change=='on'].values
        if on_offs.iloc[-1].state_change=='on':
            starts = starts[:-1]
        
        ends = on_offs.time[on_offs.state_change=='off'].values
        if on_offs.iloc[0].state_change=='off':
            ends = ends[1:]
        
        activations = pd.DataFrame({'start':starts, 'end':ends})
        if activations.shape[0] > 0:
            activations['energy'] = activations.apply(lambda r: readings[r.start:r.end].sum(),
                                                      axis=1) * self.sample_rate
        else:
            activations['energy'] = []
            
        activations = activations[activations.energy > self.min_energy]
        
        return activations
    
    def split_washingmachinetumbledrier(self, activation, tumbledrier_heating_min_off=1800,
                                   washingmachine_min_on=900):
        """split a washingmachinetumbledrier activation into washing machine and tumble drier"""
        
        assert self.appliance == 'tdheating'
        start = activation.index[0]
        end = activation.index[-1]

        heating_events = self.get_activations(activation)

        if heating_events.shape[0] > 0 and (end - heating_events.end.iloc[-1] <= 
                                dt.timedelta(seconds=tumbledrier_heating_min_off)):
            tumbledrier_start = heating_events.start.iloc[-1]
        else:
            tumbledrier_start = end

        if tumbledrier_start - start < dt.timedelta(seconds=washingmachine_min_on):
            tumbledrier_start = start

        return tumbledrier_start
    
    def split_wmtd_readings(self, readings):
        """ split readings into washing machine and tumble drier"""
        
        wmtd = 'washingmachinetumbledrier'
        assert self.appliance == wmtd
        
        if 'washingmachine' not in readings.keys():
            readings['washingmachine'] = 0
        if 'tumbledrier' not in readings.keys():
            readings['tumbledrier'] = 0
            
        activations = self.get_activations(readings[wmtd])
        activation_detector = ActivationDetector('tdheating', rulesfile=self.rulesfile)
        
        for index, row in activations.iterrows():
            activation = readings.loc[row.start:row.end, wmtd]
            td_start = activation_detector.split_washingmachinetumbledrier(activation)
            readings.loc[row.start:td_start, 'washingmachine'] += readings.loc[
                                                                       row.start:td_start, wmtd]
            readings.loc[td_start:row.end, 'tumbledrier'] += readings.loc[td_start:row.end, wmtd]
        
        return readings


if __name__ == '__main__':
    print("No main module - import ActivationDetector from activations and then call get_activations")
