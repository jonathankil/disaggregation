# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 09:52:02 2017

Create inferred BAMs for all homes. To be run every few hours.

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
from session_manager import createSession, session

import datetime
import subprocess
import numpy as np
from read_ideal_database import read_ideal_database
from ideal_data_model import PRODUCTION_KEY
from bam_writer import NilmModel

reader = read_ideal_database(PRODUCTION_KEY)
    
homes = reader.find_all_homes()

for home in homes:

    if home.securehome[0].installdate is not None:

        lastRun = reader.find_last_run(home.homeid, NilmModel.model_id)

        lastHour = datetime.datetime.utcnow()-datetime.timedelta(hours=2)
        endOfRun = lastHour.replace(minute=59, second=59, microsecond=0, tzinfo=None)

        if lastRun is None:
            startOfRun = home.securehome[0].installdate
        else:
            startOfRun = lastRun.endtime+datetime.timedelta(seconds=1)

        if startOfRun < endOfRun:
            args = [sys.executable, 'bam_writer.py', '--homeid', str(home.homeid),
                         '--starttime', startOfRun.strftime('%Y-%m-%d %H:%M:%S'),
                         '--endtime', endOfRun.strftime('%Y-%m-%d %H:%M:%S'),
                         '--environment', PRODUCTION_KEY]
            if lastRun is None:
                args.append('--alltime')

            subprocess.call(args)
