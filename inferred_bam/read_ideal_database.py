# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:26:35 2016

@author: mzhong
"""

## A script to test connection with SQLAlchemy
## to be used in IPython, defines the "session" variable that can be used in queries

## 3 magic lines to make SQLAlchemy available
import __main__; __main__.__requires__ = __requires__ = []
__requires__.append('SQLAlchemy >= 0.8.2')
import pkg_resources; pkg_resources.require(__requires__)

# Make sure that the shared library is on python path
import sys, os
lib_path = os.path.abspath('../database/python_ideal')
sys.path.append(lib_path)
sys.path.append(os.path.abspath('../data_conversion'))

from data_store import ReadingDataStore
from ideal_data_model import *
from session_manager import createSession, session, createTestSession, createIdeal1Session, createIdeal2Session, createIdeal4Session
import datetime
from threading import Thread

import pandas as pd
from pandas import HDFStore, MultiIndex
from pandas import DataFrame, Series

import numpy as np


class read_ideal_database:
    def __init__(self, environment, use_local_data=False):
        print('An instance to read data from IDEAL database')
        password = '3d21lr21d2r'
        username = 'reader'
        writePassword = 'pwd4annot'
        writeUsername = 'annot'

        if environment != PRODUCTION_KEY:
            self.mysession = createTestSession(username, password)
            self.myReadingsSession = createTestSession(username, password)
            self.writeSession = createTestSession(writeUsername, writePassword)
        else:
            self.mysession = createSession(username, password)
            self.myReadingsSession = createSession(username, password)
            self.writeSession = createSession(writeUsername, writePassword)
        self.allrawreading = {}
        self.environment = environment
        self.use_local_data = use_local_data
        
    def read_sensordata(self, start_time, end_time, sensorId):
        readings = self.find_readings_for_sensor_inclusive(start_time, end_time, sensorId)
    
        self.allrawreading[sensorId] = readings
        self.environment = self.environment

    def get_all_raw_reading(self):
        return self.allrawreading

    def find_home_by_homeid(self, homeid=0):
        mysession = self.mysession
        homes = mysession.query(Home).filter(Home.homeid==homeid)
        return homes

    def find_applianceid_by_appliancetype(self, homeid, appliancetype):
        mysession = self.mysession
        appliances = mysession.query(Appliance).filter(Appliance.homeid==homeid, Appliance.appliancetype==appliancetype)
        if appliances.count()==1:
            return appliances.first().applianceid

    def find_installed_homes(self):
        mysession = self.mysession
        homes = mysession.query(Home).join(Home.securehome).filter(Securehome.category=='installed')
        return homes

    def find_all_homes(self):
        mysession = self.mysession
        homes = mysession.query(Home).join(Home.securehome).filter(Securehome.category.in_(['installed', 'droppedout', 'uninstalled']))
        return homes

    def find_rms_ct_clamp(self, home, current_range):
        mysession = self.mysession
        sensors = mysession.query(Sensor).join(Sensor.sensorbox).join(Sensor.room).filter(Room.homeid==home.homeid, Sensor.type=='electric', Sensor.counter==2, Sensorbox.currentrange==current_range)
        return sensors
    
    def find_previous_on(self, homeid, time, appliancetype, applianceid):
        mysession = self.mysession
        # returns the last time prior to a given time that an appliance was switched on
        # Required if new run starts with an off event.
        if applianceid is not None:
            previousOn = mysession.query(Inferredapplianceonoff).filter(Inferredapplianceonoff.homeid==homeid, Inferredapplianceonoff.applianceid==applianceid, Inferredapplianceonoff.time<time, Inferredapplianceonoff.eventtype=='on').order_by(Inferredapplianceonoff.time.desc()).first()
        else:
            previousOn = mysession.query(Inferredapplianceonoff).filter(Inferredapplianceonoff.homeid==homeid, Inferredapplianceonoff.applianceid is None, Inferredapplianceonoff.appliancetype==appliancetype, Inferredapplianceonoff.time<time, Inferredapplianceonoff.eventtype=='on').order_by(Inferredapplianceonoff.time.desc()).first()
        return previousOn
    
    def find_previous_on_off(self, homeid, time, appliancetype, applianceid):
        mysession = self.mysession
        # Returns the last onOff event prior to a given time.
        # Obtained to compare last event of the previous run with the first onOff of the new run to avoid duplication.
        if applianceid is not None:
            previousOnOff = mysession.query(Inferredapplianceonoff).filter(Inferredapplianceonoff.homeid==homeid, Inferredapplianceonoff.applianceid==applianceid, Inferredapplianceonoff.time<time).order_by(Inferredapplianceonoff.time.desc()).first()
        else:
            previousOnOff = mysession.query(Inferredapplianceonoff).filter(Inferredapplianceonoff.homeid==homeid, Inferredapplianceonoff.applianceid is None, Inferredapplianceonoff.appliancetype==appliancetype, Inferredapplianceonoff.time<time).order_by(Inferredapplianceonoff.time.desc()).first()
        return previousOnOff
    
    def find_onoffs_in_period(self, homeid, start, end, appliancetype, applianceid):
        mysession = self.mysession
        # returns all the onOffs in a given period.
        # Used for obtaining onOffs detected on the previous run in the overlap period. 
        if applianceid is not None:
            onOffs = mysession.query(Inferredapplianceonoff).filter(Inferredapplianceonoff.homeid==homeid, Inferredapplianceonoff.applianceid==applianceid, Inferredapplianceonoff.time>start, Inferredapplianceonoff.time<end)
        else:
            onOffs = mysession.query(Inferredapplianceonoff).filter(Inferredapplianceonoff.homeid==homeid, Inferredapplianceonoff.applianceid is None, Inferredapplianceonoff.appliancetype==appliancetype,  Inferredapplianceonoff.time>start, Inferredapplianceonoff.time<end)
        
        time = []
        eventType = []
        result = None
        
        for onoff in onOffs:
            time.append(onoff.time)
            eventType.append(onoff.eventtype)
            result = Series(index=time, data=eventType).sort_index()
        
        return result
    
    def find_readings_for_sensor(self, start_time, end_time, sensorId):
        mysession = self.mysession
        sensor_reading = mysession.query(Reading).filter(Reading.sensorid==sensorId).\
                filter(Reading.time >= start_time).filter(Reading.time < end_time)

        sensorReadings = pd.read_sql(sensor_reading.statement, mysession.bind).sort_index()
        return sensorReadings
    
    def find_prev_reading_for_sensor(self, time, sensorId):
        mysession = self.mysession
        sensor_reading = mysession.query(Reading).filter(Reading.sensorid==sensorId).\
                filter(Reading.time < time).order_by(Reading.time.desc()).first()
        return sensor_reading
    
    def find_readings_for_sensor_inclusive(self, start_time, end_time, sensorId):
        mysession = self.myReadingsSession
        sensor_reading = mysession.query(Reading).filter(Reading.sensorid==sensorId).\
                filter(Reading.time >= start_time).filter(Reading.time <= end_time)
        sensorReadings = pd.read_sql(sensor_reading.statement, mysession.bind).sort_index()
        return sensorReadings
        
    def write_runid(self, homeid, starttime, endtime, modelid):
        mysession = self.writeSession
        inferredBamCreation = Inferredbamcreation(starttime=starttime, endtime=endtime, homeid=homeid, modelid=modelid)
        mysession.add(inferredBamCreation)
        mysession.commit()
        mysession.refresh(inferredBamCreation)
        return inferredBamCreation.runid
    
    def find_last_run(self, homeid, modelid):
        mysession = self.mysession
        run = mysession.query(Inferredbamcreation).filter(Inferredbamcreation.homeid==homeid, Inferredbamcreation.modelid==modelid).\
                order_by(Inferredbamcreation.endtime.desc()).first()
        return run
    
    def write_on_offs(self, homeid, appliancetype, applianceid, onOffs, runid):
        mysession = self.writeSession
        for index, onOff in onOffs.iterrows():
            applianceOnOff = Inferredapplianceonoff(appliancetype=appliancetype, homeid=homeid, applianceid=applianceid, time=onOff['time'], eventtype=onOff['state_change'], runid=runid)
            mysession.add(applianceOnOff)
        mysession.commit()
    
    def write_bams(self, bams, homeid, appliancetype, applianceid, runid, modelid):
        mysession = self.writeSession
        for index, bam in bams.iterrows():
            endtime = datetime.datetime.strptime(bam['start time'],'%Y-%m-%d %H:%M:%S')+datetime.timedelta(seconds=bam['duration (seconds)'])
            DBbam = Applianceuse(appliancetype=appliancetype, homeid=homeid, applianceid=applianceid,
                               starttime=bam['start time'], endtime=endtime.strftime('%Y-%m-%d %H:%M:%S'), watthoursused=bam['energy (Whs)'], type='inferred', inferredrunid=runid, modelid=modelid)
            mysession.add(DBbam)
        mysession.commit()
        
    def write_usages(self, usages):
        mysession = self.writeSession
        usages.to_sql('applianceusage', mysession.bind, if_exists='append', index=False)

        # Obtains all the readings for a given sensor (Jan 2015 to present)
    # spread between the different servers/databases
    # If adapting this to obtain readings for a given time period,
    # see Java code in models/summaries/UsageSummaryManager.getReadingsQueries()
    def find_all_time_readings_for_sensor(self, sensorId):
        if self.environment != PRODUCTION_KEY:
            sensorReadings = self.find_readings_for_sensor_inclusive(datetime.datetime.strptime('2015-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'),\
                                                           datetime.datetime.utcnow(),\
                                                           sensorId)
            return sensorReadings

        all_readings = []

        # get locally stored data
        if self.use_local_data:
            with ReadingDataStore() as s:
                if sensorId in s.get_sensorids():
                    retrieved_readings = False
                    try:
                        readings = s.get_sensor_readings(sensorId)
                        retrieved_readings = True
                    except:
                        pass

                    if readings.shape[0] > 0 and retrieved_readings:
                        all_readings.append(readings)
                        starttime = all_readings[0].iloc[-1]['time']
                        endtime = datetime.datetime.utcnow()
                        all_readings.append(self.find_sensor_readings_all_db(starttime, endtime, sensorId))
                        all_readings = pd.concat(all_readings).sort_values('time').drop_duplicates()
                        return all_readings

        password = '3d21lr21d2r'
        username = 'reader'
        ideal1Session = createIdeal1Session(username, password)
        ideal2Session = createIdeal2Session(username, password)
        ideal3Session = self.myReadingsSession
        ideal4Session = createIdeal4Session(username, password)

        # locations ordered by ascending start time.
        readingLocations = self.find_reading_locations()
        lastEnd = None
        threads = []

        for location in readingLocations:

            # Switch to correct DB via Session
            currentSession = ideal3Session
            if (location.server == 'ideal1'):
                currentSession = ideal1Session
            if (location.server == 'ideal2'):
                currentSession = ideal2Session
            if (location.server == 'ideal4'):
                currentSession = ideal4Session

            thread = ReadingQueryThread(currentSession, sensorId)
            thread.start()
            threads.append(thread)



        for thread in threads:
            readings = thread.join()
            if readings is not None:
                all_readings.append(readings)

        ideal1Session.close()
        ideal2Session.close()
        ideal4Session.close()

        all_readings = pd.concat(all_readings).sort_values('time').drop_duplicates()
        return all_readings

    def find_sensor_readings_all_db(self,  start_time, end_time, sensorId):
        """query all databases for sensor readings """
        # TODO test this
        if self.environment != PRODUCTION_KEY:
            sensorReadings = self.find_readings_for_sensor_inclusive(start_time, end_time, sensorId)
            return sensorReadings

        password = '3d21lr21d2r'
        username = 'reader'
        ideal1Session = createIdeal1Session(username, password)
        ideal2Session = createIdeal2Session(username, password)
        ideal3Session = self.myReadingsSession
        ideal4Session = createIdeal4Session(username, password)

        # locations ordered by ascending start time.
        readingLocations = ideal3Session.query(Readinglocation).order_by(Readinglocation.starttime.desc())
        lastStart = None
        sensorReadings = None

        for idx, location in enumerate(readingLocations):
            # Need to create query with appropriate end/start times for each DB

            # Deal with NULLs in readingLocations table
            # if (queryEnd is None):
            #     queryEnd = datetime.datetime.utcnow()
            queryEnd = location.endtime
            if queryEnd is None or queryEnd > end_time:
                queryEnd = end_time

            queryStart = location.starttime
            if queryStart is None or queryStart < start_time:
                queryStart = start_time

            # Deal with overlaps in reading locations. Only query time period before
            # that covered in previous query
            if lastStart is not None and queryEnd > lastStart:
                queryEnd = lastStart - datetime.timedelta(seconds=1)
            lastStart = queryStart

            if queryStart < queryEnd:

                # Switch to correct DB via Session
                currentSession = ideal3Session
                if (location.server == 'ideal1'):
                    currentSession = ideal1Session
                elif (location.server == 'ideal2'):
                    currentSession = ideal2Session
                elif (location.server == 'ideal4'):
                    currentSession = ideal4Session

                sensor_reading = currentSession.query(Reading).filter(Reading.sensorid==sensorId, Reading.time >= queryStart, Reading.time <= queryEnd)
                newSensorReadings = pd.read_sql(sensor_reading.statement, currentSession.bind)

                if sensorReadings is None:
                    sensorReadings = newSensorReadings
                else:
                    sensorReadings = newSensorReadings.append(sensorReadings)

        ideal1Session.close()
        ideal2Session.close()
        ideal4Session.close()

        sensorReadings = sensorReadings.sort_values('time').drop_duplicates()
        return sensorReadings

    def find_reading_locations(self):
        mysession = self.mysession
        locations = mysession.query(Readinglocation).order_by(Readinglocation.starttime.asc())
        return locations

class ReadingQueryThread(Thread):
    """Thread for asynchronous readings query"""
    def __init__(self, session, sensorId):
        Thread.__init__(self)
        self.session = session
        self.sensorId = sensorId
        self._return = None

    def run(self):
        sensor_reading = self.session.query(Reading).filter(Reading.sensorid==self.sensorId)
        self._return = pd.read_sql(sensor_reading.statement, self.session.bind)

    def join(self, timeout=None):
        Thread.join(self, timeout)
        return self._return
