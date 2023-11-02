# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:29:29 2017

@author: edmun
"""

## 3 magic lines to make SQLAlchemy available
import __main__; __main__.__requires__ = __requires__ = []
__requires__.append('SQLAlchemy >= 0.8.2')
import pkg_resources; pkg_resources.require(__requires__)

# Make sure that the shared library is on python path
import sys, os
lib_path = os.path.abspath('../../database/python_ideal')
sys.path.append(lib_path)
lib_path = os.path.abspath('')
sys.path.append(lib_path)

from ideal_data_model import *
from session_manager import createTestSession, session
import datetime

import pandas as pd
from pandas import HDFStore, MultiIndex
from pandas import DataFrame, Series

import subprocess

def date_of_days_past(daysPast):
    pastDate = datetime.datetime.utcnow()-datetime.timedelta(days=daysPast);
    return pastDate.strftime('%Y-%m-%d')
    
def write_readings(writeSession):
    writeSession.query(Reading).filter(Reading.sensorid.in_([1647, 1648, 1649])).delete(synchronize_session='fetch')
    writeSession.query(Applianceuse).filter(Applianceuse.applianceid.in_([1306, 1307, 1308])).delete(synchronize_session='fetch')
    writeSession.query(Applianceonoff).filter(Applianceonoff.applianceid.in_([1306, 1307, 1308])).delete(synchronize_session='fetch')
    
    y = date_of_days_past(1)
    y2 = date_of_days_past(2)
    y3 = date_of_days_past(3)
    y4 = date_of_days_past(4)
    y5 = date_of_days_past(5)
    y6 = date_of_days_past(6)
    
    # 10->10:03 1800Wh
    writeSession.add(Reading(sensorid=1647, time=y+' 10:00:00', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y+' 10:03:00', value=0))
    # 15->15:10 12000Wh
    writeSession.add(Reading(sensorid=1647, time=y+' 15:00:00', value=72000))
    writeSession.add(Reading(sensorid=1647, time=y+' 15:10:00', value=10))
    # 17:03->17:08 Lots of repeats 60+120+60+180+240=6600Wh
    writeSession.add(Reading(sensorid=1647, time=y+' 17:03:00', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y+' 17:04:00', value=72000))
    writeSession.add(Reading(sensorid=1647, time=y+' 17:05:00', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y+' 17:05:30', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y+' 17:06:00', value=108000))
    writeSession.add(Reading(sensorid=1647, time=y+' 17:07:00', value=144000))
    writeSession.add(Reading(sensorid=1647, time=y+' 17:08:00', value=0))
    writeSession.add(Reading(sensorid=1647, time=y+' 17:09:00', value=10))
    writeSession.add(Reading(sensorid=1647, time=y+' 17:10:00', value=10))
    writeSession.add(Reading(sensorid=1647, time=y+' 17:11:00', value=10))
    writeSession.add(Reading(sensorid=1647, time=y+' 17:12:00', value=0))
    
    # On but no off in sample period. No BAM
    writeSession.add(Reading(sensorid=1647, time=y+' 23:30:00', value=36000))
    # 03:00:00-> On with no off - should be ignored.
    writeSession.add(Reading(sensorid=1648, time=y+' 03:00:00', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y+' 03:05:00', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y+' 03:10:00', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y+' 04:00:00', value=36000))
    # 02:00:00-> Off with no on - should be ignored.
    writeSession.add(Reading(sensorid=1648, time=y+' 02:00:00', value=10))
    writeSession.add(Reading(sensorid=1648, time=y+' 02:10:00', value=10))
    writeSession.add(Reading(sensorid=1648, time=y+' 02:40:00', value=10))
    writeSession.add(Reading(sensorid=1648, time=y+' 02:50:00', value=10))
    # 10:00:45->10:03:45. Varying power. 270Wh
    writeSession.add(Reading(sensorid=1648, time=y+' 10:00:45', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y+' 10:01:45', value=72000))
    writeSession.add(Reading(sensorid=1648, time=y+' 10:02:45', value=54000))
    writeSession.add(Reading(sensorid=1648, time=y+' 10:03:45', value=10))
    # 15:10:00->15:00:12 Series of short ons and offs. 70Wh
    writeSession.add(Reading(sensorid=1649, time=y+' 15:10:00', value=36000))
    writeSession.add(Reading(sensorid=1649, time=y+' 15:10:02', value=0))
    writeSession.add(Reading(sensorid=1649, time=y+' 15:10:06', value=36000))
    writeSession.add(Reading(sensorid=1649, time=y+' 15:10:09', value=0))
    writeSession.add(Reading(sensorid=1649, time=y+' 15:10:10', value=36000))
    writeSession.add(Reading(sensorid=1649, time=y+' 15:10:12', value=0))
    # Prev 9:10->9:15 3000Wh
    writeSession.add(Reading(sensorid=1647, time=y2+' 09:10:00', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y2+' 09:15:00', value=0))
    # 23:59:57->00:02:00 On just before cutoff. 123Wh
    writeSession.add(Reading(sensorid=1648, time=y2+' 23:59:57', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y+' 00:02:00', value=0))
    # Short on in overlap period. No events or BAMs.
    writeSession.add(Reading(sensorid=1647, time=y2+' 23:59:45', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y2+' 23:59:53', value=10))
    writeSession.add(Reading(sensorid=1647, time=y+' 00:02:00', value=0))
    # Prev 23:59:45->23:59:57 On and off in overlap period. 120Wh
    writeSession.add(Reading(sensorid=1649, time=y2+' 23:59:45', value=36000))
    writeSession.add(Reading(sensorid=1649, time=y2+' 23:59:57', value=10))
    writeSession.add(Reading(sensorid=1649, time=y+' 00:02:00', value=0))
    # Furth 23:59:35->23:59:58 On over overlap cutoff. 23Wh
    writeSession.add(Reading(sensorid=1648, time=y3+' 23:59:35', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y3+' 23:59:58', value=10))
    writeSession.add(Reading(sensorid=1648, time=y2+' 00:02:00', value=0))
    # Spurious readings. No BAM
    writeSession.add(Reading(sensorid=1648, time=y3+' 10:59:05', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y3+' 15:59:00', value=0))
    writeSession.add(Reading(sensorid=1648, time=y3+' 15:59:13', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y3+' 17:59:15', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y3+' 17:59:55', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y3+' 18:03:55', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y3+' 20:59:15', value=0))
    
    
    # Furth 23:59:05->23:59:37 short on over overlap, short off, long on 300Wh
    writeSession.add(Reading(sensorid=1647, time=y3+' 23:59:05', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y3+' 23:59:13', value=10))
    writeSession.add(Reading(sensorid=1647, time=y3+' 23:59:15', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y3+' 23:59:37', value=10))
    writeSession.add(Reading(sensorid=1647, time=y2+' 00:02:00', value=0))
    # EvenFurth 23:59:06->23:59:29 short on either side of over overlap cutoff 210Wh
    writeSession.add(Reading(sensorid=1647, time=y4+' 23:59:06', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y4+' 23:59:13', value=10))
    writeSession.add(Reading(sensorid=1647, time=y4+' 23:59:15', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y4+' 23:59:29', value=10))    
    # EvenFurth 13:02:00->07:15:00 On for more than 24 hours - 42hrs 13m - 1519800Wh
    # Update - should just be ignored
    writeSession.add(Reading(sensorid=1649, time=y4+' 13:02:00', value=36000))
    writeSession.add(Reading(sensorid=1649, time=y2+' 07:15:00', value=10))    
    # EvenFurth 23:59:37->23:59:47 On over overlap period, short either side. 10Wh
    writeSession.add(Reading(sensorid=1648, time=y4+' 23:59:37', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y4+' 23:59:47', value=10))
    # Long ago! 11:00:00->11:08:00 Normal but long ago with clear day afterwards 
    writeSession.add(Reading(sensorid=1648, time=y6+' 11:00:00', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y6+' 11:08:00', value=10))
    # 23:57:00->00:00:00 Reading on midnight 1800Wh
    writeSession.add(Reading(sensorid=1647, time=y5+' 23:57:00', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y4+' 00:00:00', value=10))
    # Long ago! 12:17:00-12:20:00 Unended on in overlap and before 1800Wh
    writeSession.add(Reading(sensorid=1649, time=y6+' 23:50:42', value=36000))
    writeSession.add(Reading(sensorid=1649, time=y6+' 23:59:42', value=36000))
    writeSession.add(Reading(sensorid=1649, time=y5+' 12:17:00', value=36000))    
    writeSession.add(Reading(sensorid=1649, time=y5+' 12:20:00', value=10))    
    # Long ago! 23:59:45-23:59:57 On briefly. Extra offs, unmatched on. 220Wh
    writeSession.add(Reading(sensorid=1647, time=y6+' 23:59:35', value=36000))
    writeSession.add(Reading(sensorid=1647, time=y6+' 23:59:57', value=0))
    writeSession.add(Reading(sensorid=1647, time=y5+' 00:00:10', value=0))
    writeSession.add(Reading(sensorid=1647, time=y5+' 00:00:30', value=0))
    writeSession.add(Reading(sensorid=1647, time=y5+' 00:01:10', value=36000))    
    # Long ago! 23:59:00->00:00:10 Short off over period cutoff, 11 
    writeSession.add(Reading(sensorid=1648, time=y5+' 23:59:50', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y5+' 23:59:57', value=10))
    writeSession.add(Reading(sensorid=1648, time=y4+' 00:00:06', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y4+' 00:00:10', value=10))
    # Long ago! On with an off too long afterwards. No BAM. 
    writeSession.add(Reading(sensorid=1648, time=y6+' 07:00:00', value=36000))
    writeSession.add(Reading(sensorid=1648, time=y6+' 10:30:00', value=10))
    
    
    writeSession.commit()
    
def find_bams(mysession, homeid, appliancety2e, applianceid):
    # returns all the bams for a given appliance.
    if applianceid is not None:
        bams = mysession.query(Applianceuse).filter(Applianceuse.applianceid==applianceid).order_by(Applianceuse.starttime.desc())
    else:
        bams = mysession.query(Applianceuse).filter(Applianceuse.applianceid is None, Applianceuse.appliancety2e==appliancety2e).order_by(Applianceuse.starttime.desc())
    
    return bams

def find_onOffs(mysession):
    homeid = 132
    appIds = [1308, 1307, 1306]
    
    for applianceid in appIds: 
        print (applianceid)
        # returns all the bams for a given appliance.
        if applianceid is not None:
            onOffs = mysession.query(Applianceonoff).filter(Applianceonoff.applianceid==applianceid).order_by(Applianceonoff.time.desc())
        else:
            onOffs = mysession.query(Applianceonoff).filter(Applianceonoff.applianceid is None, Applianceonoff.appliancety2e==appliancety2e).order_by(Applianceonoff.time.desc())
        
        for onOff in onOffs:
            print (onOff.time, onOff.eventty2e)
        
    return onOffs


def check_bams(mysession):
    y = date_of_days_past(1)
    y2 = date_of_days_past(2)
    y3 = date_of_days_past(3)
    y4 = date_of_days_past(4)
    y5 = date_of_days_past(5)
    y6 = date_of_days_past(6)
    appIds = [1308, 1307, 1306]
    bamCounts = [8, 6, 3]
    allStarts = []
    allEnds = []
    allValues = []
    allStarts.insert(0, [y+' 17:03:00', y+' 15:00:00', y+' 10:00:00', y2+' 09:10:00', y3+' 23:59:05', y4+' 23:59:06', y5+' 23:57:00', y6+' 23:59:35'])
    allEnds.insert(0, [y+' 17:08:00', y+' 15:10:00', y+' 10:03:00', y2+' 09:15:00', y3+' 23:59:37', y4+' 23:59:29', y4+' 00:00:00', y6+' 23:59:57'])
    allValues.insert(0, [6600, 12000, 1800, 3000, 300, 210, 1800, 220])
    allStarts.insert(1, [y+' 10:00:45', y2+' 23:59:57', y3+' 23:59:35', y4+' 23:59:37', y5+' 23:59:50', y6+' 11:00:00'])
    allEnds.insert(1, [y+' 10:03:45', y+' 00:02:00', y3+' 23:59:58', y4+' 23:59:47', y4+' 00:00:10', y6+' 11:08:00'])
    allValues.insert(1, [270, 123, 23, 10, 11, 480])
    allStarts.insert(2, [y+' 15:10:00', y2+' 23:59:45', y5+' 12:17:00'])
    allEnds.insert(2, [y+' 15:10:12', y2+' 23:59:57', y5+' 12:20:00'])
    allValues.insert(2, [70, 120, 1800])
    appCount = 0
    for appId in appIds:
        bams = find_bams(mysession, 132, None, appId)
        bamCount = 0
        starts = allStarts[appCount]
        ends = allEnds[appCount]
        values = allValues[appCount]
        for bam in bams:
            assert str(bam.starttime) == str(starts[bamCount]), "%s Wrong start time: %s. Should be %s" % (appId, bam.starttime, starts[bamCount])  
            assert str(bam.endtime) == str(ends[bamCount]), "%s Wrong end time: %s. Should be %s" % (appId, bam.endtime, ends[bamCount])  
            assert bam.watthoursused == values[bamCount], "%s Wrong value: %s. Should be %s" % (appId, bam.watthoursused, values[bamCount])  
            bamCount += 1
        assert bamCount == bamCounts[appCount], "Wrong number of BAMS for %r: %r" % (appId, bamCount)        
        appCount += 1
    
    return bams


password = '3d21lr21d2r'
username = 'reader'  
mysession = createTestSession(username, password)

writePassword = 'pwd4annot'
writeUsername = 'annot'
writeSession = createTestSession(writeUsername, writePassword)

write_readings(writeSession)
subprocess.call([sys.executable, 'bam_writer.py', '--homeid', '132', '--starttime', date_of_days_past(6)+' 00:00:00', '--endtime', date_of_days_past(6)+' 23:59:59', '--environment', 'test'])
subprocess.call([sys.executable, 'bam_writer.py', '--homeid', '132', '--starttime', date_of_days_past(5)+' 00:00:00', '--endtime', date_of_days_past(5)+' 23:59:59', '--environment', 'test'])
subprocess.call([sys.executable, 'bam_writer.py', '--homeid', '132', '--starttime', date_of_days_past(4)+' 00:00:00', '--endtime', date_of_days_past(4)+' 23:59:59', '--environment', 'test'])
subprocess.call([sys.executable, 'bam_writer.py', '--homeid', '132', '--starttime', date_of_days_past(3)+' 00:00:00', '--endtime', date_of_days_past(3)+' 23:59:59', '--environment', 'test'])
subprocess.call([sys.executable, 'bam_writer.py', '--homeid', '132', '--starttime', date_of_days_past(2)+' 00:00:00', '--endtime', date_of_days_past(2)+' 23:59:59', '--environment', 'test'])
subprocess.call([sys.executable, 'bam_writer.py', '--homeid', '132', '--starttime', date_of_days_past(1)+' 00:00:00', '--endtime', date_of_days_past(1)+' 23:59:59', '--environment', 'test'])

check_bams(mysession)
#find_onOffs()
#==============================================================================
# try:
#     subprocess.check_output([sys.executable, 'bam_writer.py'], stderr=subprocess.STDOUT, shell=True)
# except subprocess.CalledProcessError as e:
#     raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
# 
#==============================================================================
