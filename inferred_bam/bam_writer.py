
# -*- coding: utf-8 -*-
"""
Created on Tue May 2nd 2017

@author: jonathan
"""

# This script puts a framework around Mingjun's Best Available Measure
# (BAM) code so that it can be run on all gold homes, writing BAMS to
# the database. This is designed to be run daily.

# 1. read a set of rules that define how to extract BAMs from appliances
# 2. From the homes passed as an argument or for all gold homes
#    (default), apply all rules that match an appliance in the home
# 3. Write the BAMs to the database


## Make SQLAlchemy available
import __main__; __main__.__requires__ = __requires__ = []
__requires__.append('SQLAlchemy >= 0.8.2')
import pkg_resources; pkg_resources.require(__requires__)
import pandas as pd
import numpy as np

# Make sure that the shared library is on python path
import sys, os
lib_path = os.path.abspath('../../database/python_ideal')
sys.path.append(lib_path)
sys.path.insert(0, '../data_conversion')

from read_ideal_database import read_ideal_database
from ideal_data_model import PRODUCTION_KEY
from electric import submeter
from nilm_fully_conv_masked import NilmFullyConvMasked as NilmModel
from data_preprocessing import ElecPreprocessor

import argparse
import logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
import json
import datetime
import re
import time

def valid_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def start_of_yesterday():
    yesterday = datetime.datetime.utcnow()-datetime.timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')+' 00:00:00'

def end_of_yesterday():
    yesterday = datetime.datetime.utcnow()-datetime.timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')+' 23:59:59'

# process command line arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='Extract possible \
                                     best available measures(BAM).')
    parser.add_argument('--homeid',
                        type=str,
                        default='0',
                        help='Provide a homeid or "all" for all homes.')
    parser.add_argument('--rules',
                        type=str,
                        default='rules.json',
                        help='Provide a JSON format rules file (defaults to rules.json')
    parser.add_argument('--starttime', 
                        type=valid_date,
                        default=start_of_yesterday(),
                        #default="2017-06-04 00:00:00",
                        help='The Start Date - format YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--endtime', 
                        type=valid_date, 
                        default=end_of_yesterday(),
                        #default="2017-06-30 23:59:59",
                        help='The End Date - format YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--environment', 
                        type=str, 
                        default='Test',
                        help='Environment - ' + PRODUCTION_KEY + ' for production')
    parser.add_argument('--modelpath',
                        type=str,
                        default='models/',
                        help='path to directory where ML models are stored')
    parser.add_argument('--maxrundays',
                        type=int,
                        default=7,
                        help='maximum number of days to use in each run, longer runs are split into multiple runs')
    parser.add_argument('--alltime', action='store_true',
                        help='query readings from all time - WARNING: may cause excessive memory usage')
    parser.add_argument('--localdata', action='store_true',
                        help='use locally stored sensor readings')
    return parser.parse_args()


def create_onoffs(reader, appliance_readings, rules, home, appliance_name, applianceid, start, end, runid):
    for rule in rules:
        # match appliance
        if (rule["appliance"]):
            pattern = re.compile(rule["appliance"])
            if not (pattern.match(appliance_name) is None):
                # Get onOffs for current appliance using given rule from file
                # and only considering the given time period.
                onOffs = apply_rule(reader, appliance_readings, rule, home, appliance_name, applianceid, start, end)
                if onOffs.empty:
                    return
                
                # Remove duplicates before writing - these will be from the
                # overlap period and already in the database.
                # We don't need them for creating BAMS. Anything but a final on will
                # have been dealt with in the previous run. We'll pick the on up again
                # later when we discover we need it.
                onOffs.drop_duplicates(keep=False, inplace=True)
                # Write freshly created onOffs to database
                reader.write_on_offs(home.homeid, appliance_name, applianceid, onOffs, runid)
                
                # Now we've written all the new onOffs we need to sanitise the DataFrame
                # so we can create BAMs. We need to start with an on and end with an off.
                # Get actual first event now any false event has been discarded.
                firstEvent = onOffs.iloc[0, onOffs.columns.get_loc('state_change')]
                firstTime = onOffs.iloc[0, onOffs.columns.get_loc('time')]
                # If the first event is an off, we need to find a matching on
                # from before. Fetch most recent one from before first event.
                if firstEvent == 'off':               
                    lastOnPrevRun = reader.find_previous_on(home.homeid, firstTime, appliance_name, applianceid)
                    # If there isn't previous data or too long ago, discard off
                    if lastOnPrevRun is not None:
                        if (firstTime - lastOnPrevRun.time) < datetime.timedelta(seconds=int(rule["max_on_duration"])):
                            times = [lastOnPrevRun.time]
                            # Add last on of previous run to the list of onOffs to be returned for creating BAMs.
                            values = ['on']
                            lastOnDF = pd.DataFrame({'time':times, 'state_change':values})
                            onOffs = pd.concat([lastOnDF, onOffs])
                        else:
                            onOffs.drop(onOffs.index[:1], inplace=True)    
                    else:
                        onOffs.drop(onOffs.index[:1], inplace=True)
                  
                if onOffs.empty:
                    return
                
                # If the onOffs end with an on, discard it. Will be picked up on the next run.
                onOffs = onOffs.reset_index()
                
                lastEvent = onOffs.iloc[-1, onOffs.columns.get_loc('state_change')]
                if lastEvent == 'on':
                    onOffs.drop(onOffs.index[-1:], inplace=True)
                
                return onOffs


def apply_rule(reader, appliance_readings, rule, home, appliancetype, applianceid, start, end):

    # Need to go back into the period of the previous run slightly
    # to catch events that straddle two different runs. Max amount of time is
    # min_off + min_on. This is where an on occurs for nearly min_on and then an off
    # for nearly min_off - should count as an on as below threshold offs are discarded.
    # We could have caught offs or ons in this overlap period already. Will need to check for duplicates before writing.
    overlapStart = appliance_readings.index[0]

    read = submeter()

    # Determine on and offs from readings.
    lastEventOfPrev = reader.find_previous_on_off(home.homeid, overlapStart, appliancetype, applianceid)

    results = read.get_ons_and_offs(
        appliance_readings,
        end,
        lastEventOfPrev,
        min_off_duration=int(rule["min_off_duration"]),
        min_on_duration=int(rule["min_on_duration"]),
        max_on_duration=int(rule["max_on_duration"]),
        on_power_threshold=int(rule["on_power_threshold"]))

    # Get onOffs recorded in database from overlap period with previous run.
    prevOnOffs = reader.find_onoffs_in_period(home.homeid, overlapStart, start, appliancetype, applianceid)

    prevDF = None
    # # Merge new onOffs with previous onOffs.
    if prevOnOffs is not None and not prevOnOffs.empty:
        prevDF = pd.DataFrame({'time':prevOnOffs.index, 'state_change':prevOnOffs.values})
        # remove events that occur before prevOnOffs
        results = results[results.time > prevDF.time.iloc[-1]]
        # remove first event from results if it is of the same type as last previous event
        if not results.empty and results.iloc[0].state_change == prevDF.state_change.iloc[-1]:
            results = results.drop(results.index[0])

    return results

    # Create the BAMs from a DataFrame of onOff events        
def create_bams(onOffs, reader, appliance_readings, home, appliancetype, applianceid, runid, rules):

    # We need all the readings as well as the onOff events in
    # order to calculate the energy use.
    readingSeries = appliance_readings.copy()
    bams = None
    
    if (onOffs is not None) and (not onOffs.empty):
        read = submeter()

        # Convert onOffs and raw readings into BAMs
        bams = read.get_bams_from_on_offs(onOffs, readingSeries, power_unit=1)

        # remove BAMS which have less than min_energy
        min_energy = 0
        for rule in rules:
        # match appliance
            if rule["appliance"]:
                pattern = re.compile(rule["appliance"])
                if not (pattern.match(appliancetype) is None):
                    if 'min_energy' in rule:
                        min_energy = int(rule['min_energy']) / 3600.0  # convert joules to Whs
                        break
        bams = bams[bams['energy (Whs)'] > min_energy]
        
    if (bams is None) or (bams.empty):
        logging.debug('No activations for '+str(home.homeid) + "'s "+ appliancetype)
    else:
        logging.debug('Found activations for '+str(home.homeid) + "'s "+ appliancetype)
        # Write to database.
        reader.write_bams(bams, home.homeid, appliancetype, applianceid, runid, NilmModel.model_id)

def get_aggregate_sensorid(reader, home, current_range, unusable_sensors):
    sensors = reader.find_rms_ct_clamp(home, current_range=current_range)
    if sensors.count() == 1:
        sensorid = sensors.first().sensorid
        if sensorid not in unusable_sensors:
            return sensorid

def get_aggregate_readings(reader, home, preprocessor, unusable_sensors, start_time, end_time, rules, nilm_appliances,
                           model, alltime):
    """query and preprocess mains electricity readings"""

    # calculate needed start and end times to account for BAM overlap and S2P offset

    max_overlap_time = 0
    for rule in rules['rules']:
        # match appliance
        if (rule["appliance"]):
            pattern = re.compile(rule["appliance"])
            for appliance_name in nilm_appliances:
                if not (pattern.match(appliance_name) is None):
                    max_overlap_time = max(max_overlap_time,
                                           int(rule["max_on_duration"]) + int(rule["min_off_duration"]))
    max_overlap_time = max(max_overlap_time, 3600)  # minimum overlap of 1 hour

    model_offset_time = model.offset*model.sample_rate
    aggregate_start = start_time - datetime.timedelta(seconds=model_offset_time + max_overlap_time)
    aggregate_end = end_time + datetime.timedelta(seconds=model_offset_time)

    sensorid_30A, sensorid_100A = [get_aggregate_sensorid(reader, home, current_range, unusable_sensors)
                                   for current_range in ['30A', '100A']]

    dummy_readings = pd.DataFrame(
    columns=['time','value','tenths_seconds_since_last_reading'])
    dummy_readings['time'] = dummy_readings['time'].astype('datetime64[ns]')

    readings_30A, readings_100A = [
        reader.find_all_time_readings_for_sensor(sensorid) if alltime
        else (reader.find_sensor_readings_all_db(aggregate_start, aggregate_end, sensorid)
              if aggregate_start < datetime.datetime.utcnow() - datetime.timedelta(days=6)
              else reader.find_readings_for_sensor_inclusive(aggregate_start, aggregate_end, sensorid))
        for sensorid in [sensorid_30A, sensorid_100A]]

    aggregate_readings = preprocessor.process_mains_clamp(readings_30A, readings_100A)

    return aggregate_readings[aggregate_start:aggregate_end]

def get_inferred_appliance_readings(reader, home, preprocessor, unusable_sensors, starttime, endtime, rules, model,
                                    alltime):
    """get inferred appliance power usage using model"""

    home_appliances = [a.appliancetype.lower() for a in home.appliances]

    # electrichob and electricoven are merged into one appliancetype "electriccooker"
    for i, appliancetype in enumerate(home_appliances):
        if appliancetype in ['electrichob', 'electricoven']:
            home_appliances[i] = 'electriccooker'

    nilm_appliances = [a for a in model.supported_appliances if a in home_appliances]

    aggregate_readings = get_aggregate_readings(reader, home, preprocessor, unusable_sensors,
                                                starttime, endtime, rules, nilm_appliances, model, alltime)

    if not aggregate_readings.empty:

        appliance_readings = model.predict(aggregate_readings, nilm_appliances)

        # rename electriccooker back to electrichob so that it fits set of standard appliancetypes
        appliance_readings = appliance_readings.rename(columns={'electriccooker':'electricoven'})
        return appliance_readings

def create_usages(reader, home, appliancetype, applianceid, appliance_readings, starttime, endtime, runid):

    # round start and end times downwards to nearest hour
    endtime = endtime + datetime.timedelta(seconds=1)
    starttime, endtime = [t.replace(minute=0, second=0, microsecond=0, tzinfo=None) for t in [starttime, endtime]]
    endtime = endtime - datetime.timedelta(seconds=1)

    usages = appliance_readings[starttime:endtime].resample('1H').sum().to_frame()
    usages = usages * NilmModel.sample_rate / 3600.0  # convert to Whs
    usages = usages.rename(columns={appliancetype: 'watthoursused'})
    usages['starttime'] = usages.index.to_series()
    usages['endtime'] = usages['starttime'] + datetime.timedelta(minutes=59, seconds=59)
    usages = usages[usages.watthoursused >= 0.5]

    usages['homeid'] = home.homeid
    usages['applianceid'] = applianceid
    usages['appliancetype'] = appliancetype
    usages['type'] = 'inferred'
    usages['inferredrunid'] = runid
    usages['modelid'] = NilmModel.model_id

    reader.write_usages(usages)


def process_home(reader, home, starttime, endtime, preprocessor, unusable_sensors, rules, model, alltime):
    """Create BAMs and applianceuse for a home"""

    # get the inferred appliance readings
    appliance_readings = get_inferred_appliance_readings(reader, home, preprocessor, unusable_sensors, starttime,
                                                         endtime, rules, model, alltime)

    # Create and record identifier for this run of BAM creation.
    # Appended to all BAMs.
    runid = reader.write_runid(home.homeid, starttime, endtime, NilmModel.model_id)

    if (appliance_readings is not None) and (not appliance_readings.empty):

        for appliancetype in appliance_readings:
            applianceid = reader.find_applianceid_by_appliancetype(home.homeid, appliancetype)

            create_usages(reader, home, appliancetype, applianceid, appliance_readings[appliancetype],
                         starttime, endtime, runid)

            onoffs = create_onoffs(reader, appliance_readings[appliancetype], rules['rules'], home, appliancetype,
                                   applianceid, starttime, endtime, runid)

            if onoffs is not None:
                create_bams(onoffs, reader, appliance_readings[appliancetype], home, appliancetype, applianceid,
                            runid, rules['rules'])

def main():
    # get the arguments
    args = get_arguments()
    
    homeid = args.homeid
    rulesfile = args.rules
    starttime = args.starttime
    endtime = args.endtime
    environment = args.environment
    modelpath = args.modelpath
    maxrundays = args.maxrundays
    alltime = args.alltime
    use_local_data = args.localdata

    with open(rulesfile) as data_file:    
        rules = json.load(data_file)

    unusable_sensors = pd.read_csv('../data_conversion/anomalous_sensors.csv', dtype={'homeid':np.int32,
                                                           'sensorid':np.int32, 'notes':str}).sensorid.values
    # define an instance
    reader = read_ideal_database(environment, use_local_data)
    # if we don't have a homeid, find all the installed homes
    
    if homeid != 'all':
        try:
            homeid = int(homeid)
        except:
            homeid = 0
        homes = reader.find_home_by_homeid(homeid)
        #logging.debug('Home ID: '+str(homeid))
    else:
        homes = reader.find_all_homes()

    # load disaggregation model and create electrical readings preprocessor
    model = NilmModel(modelpath)
    preprocessor = ElecPreprocessor(sample_rate=model.sample_rate)

    max_run_duration = datetime.timedelta(days=maxrundays)

    for home in homes:
        logging.debug('Process home:' + str(home.homeid)+'.')

        if alltime:
            process_home(reader, home, starttime, endtime, preprocessor, unusable_sensors, rules, model, alltime)
        else:
            # split run into chunks of max 1 week length to constrain memory usage
            chunk_start = starttime
            while chunk_start < endtime:
                chunk_end = min(endtime, chunk_start + max_run_duration - datetime.timedelta(seconds=1))
                process_home(reader, home, chunk_start, chunk_end, preprocessor, unusable_sensors, rules, model, alltime)
                chunk_start = chunk_end + datetime.timedelta(seconds=1)



if __name__ == '__main__':
    df_bams=main()


