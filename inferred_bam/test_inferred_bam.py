"""
Created on 22/06/18
@author Cillian Brewitt
"""

## 3 magic lines to make SQLAlchemy available
import __main__; __main__.__requires__ = __requires__ = []
__requires__.append('SQLAlchemy >= 0.8.2')
import pkg_resources; pkg_resources.require(__requires__)

import sys
sys.path.insert(0, '../../database/python_ideal')

from ideal_data_model import *
from session_manager import createTestSession, session
import datetime as dt
import subprocess
import pandas as pd
import pytest


def delete_readings(writeSession, sensorids):
    # delete existing readings
    writeSession.query(Reading).filter(Reading.sensorid.in_(sensorids)).delete(synchronize_session='fetch')

def delete_appliance_use(writeSession):
    applianceid = 1309
    writeSession.query(Applianceuse).filter(Applianceuse.applianceid==applianceid).delete(synchronize_session='fetch')
    writeSession.query(Inferredapplianceonoff).filter(Inferredapplianceonoff.applianceid==applianceid).delete(synchronize_session='fetch')

def write_sythetic_readings(writeSession, bams_to_write):
    homeid = 141
    sensorid_30A = 1135
    sensorid_100A = 1133

    delete_readings(writeSession, [sensorid_30A, sensorid_100A])

    readings = pd.DataFrame({'time': pd.date_range('2018-06-20', '2018-06-24', freq='S'),
                             'value': 0, 'tenths_seconds_since_last_reading': 10, 'sensorid': sensorid_30A})
    readings = readings.set_index('time').sort_index()

    for idx, bam in bams_to_write.iterrows():
        readings.loc[bam.starttime:bam.endtime, 'value'] = 2500

    readings.to_sql('reading', writeSession.bind, if_exists='append')

def find_bams(mySession, applianceid):
    query = mySession.query(Applianceuse).filter(Applianceuse.applianceid==applianceid).order_by(Applianceuse.starttime.desc())
    return pd.read_sql(query.statement, mySession.bind)

def assert_approx_equal(a, b, delta):
    assert abs(a - b) <= delta, 'assertion {0} ~ {1} failed'.format(a, b)

def check_bams(mySession, expected_bams):
    bams = find_bams(mySession, 1309)

    for idx, expected_bam in expected_bams.iterrows():
        assert_approx_equal(expected_bam.starttime, bams.iloc[idx].starttime, dt.timedelta(seconds=20))
        assert_approx_equal(expected_bam.endtime, bams.iloc[idx].endtime, dt.timedelta(seconds=20))
        assert_approx_equal(expected_bam.watthoursused, bams.iloc[idx].watthoursused, 20)

    assert bams.shape[0] == expected_bams.shape[0]

def kettle_bams_from_bam_times(bam_times):
    bams = pd.DataFrame({'starttime': pd.to_datetime([t[0] for t in bam_times]),
                         'endtime': pd.to_datetime([t[1] for t in bam_times])})

    # assume constant power of 2.5 kW
    bams['watthoursused'] = (bams.endtime - bams.starttime).apply(dt.timedelta.total_seconds) * 2500 / 3600
    bams = bams.sort_values('starttime')
    return bams

def test_sythetic_readings():

    bam_times = (('2018-06-22 23:55:00', '2018-06-22 23:56:00'),  # usage before overlap with time offset
                 ('2018-06-22 08:59:00', '2018-06-22 09:01:00'),  # ordinary usage
                 ('2018-06-21 23:59:00', '2018-06-22 00:01:00'),  # usage on overlap
                 ('2018-06-21 01:00:00', '2018-06-21 01:01:00'),  # ordinary usage
                 ('2018-06-20 23:57:00', '2018-06-20 23:59:00'),  # usage before overlap
                 )

    bams = kettle_bams_from_bam_times(bam_times)

    writePassword = 'pwd4annot'
    writeUsername = 'annot'
    writeSession = createTestSession(writeUsername, writePassword)
    delete_appliance_use(writeSession)
    write_sythetic_readings(writeSession, bams)

    # create BAMs
    subprocess.call([sys.executable, 'bam_writer.py', '--homeid', '141', '--starttime', '2018-06-21 00:00:00', '--endtime', '2018-06-22 00:00:00', '--environment', 'test'])
    subprocess.call([sys.executable, 'bam_writer.py', '--homeid', '141', '--starttime', '2018-06-22 00:00:00', '--endtime', '2018-06-23 00:00:00', '--environment', 'test'])
    subprocess.call([sys.executable, 'bam_writer.py', '--homeid', '141', '--starttime', '2018-06-22 23:59:59', '--endtime', '2018-06-24 00:00:00', '--environment', 'test'])  # start offset by 1 second

    check_bams(writeSession, bams)

def test_jittered_readings():
    # check if bam detection is robust to jittered readings between runs when handling duplicate BAMS
    applianceid = 1309

    writePassword = 'pwd4annot'
    writeUsername = 'annot'
    writeSession = createTestSession(writeUsername, writePassword)

    bam_times = (('2018-06-22 23:55:00', '2018-06-22 23:56:00'),)
    bams = kettle_bams_from_bam_times(bam_times)
    delete_appliance_use(writeSession)
    write_sythetic_readings(writeSession, bams)

    subprocess.call([sys.executable, 'bam_writer.py', '--homeid', '141', '--starttime', '2018-06-22 00:00:00', '--endtime', '2018-06-23 00:00:00', '--environment', 'test'])

    check_bams(writeSession, bams)

    # write jittered readings
    bam_times = (('2018-06-22 23:55:10', '2018-06-22 23:56:10'),)
    bams = kettle_bams_from_bam_times(bam_times)
    write_sythetic_readings(writeSession, bams)

    subprocess.call([sys.executable, 'bam_writer.py', '--homeid', '141', '--starttime', '2018-06-23 00:00:00', '--endtime', '2018-06-24 00:00:00', '--environment', 'test'])

    check_bams(writeSession, bams)

if __name__ == '__main__':
    pytest.main()
    #test_sythetic_readings()
    #test_jittered_readings()
