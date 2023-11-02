import requests
from datetime import datetime,timedelta
import argparse
import pandas as pd
import json
import sys
import sqlite3
import os
import gzip
from LiteNILM import NILM_Model
import re
sys.path.insert(0, os.path.dirname(__file__) + './inferred_bam')
from electric import submeter
from activations import ActivationDetector
import math
import pytz
utc = pytz.UTC

# grab recent CAD / Clip data from local DB and pass it through the NILM processes to get recent activations.
# make the results available to rules / anomaly detection processes

# Handle command line arguments
parser = argparse.ArgumentParser(description='Disaggregate 10 second Peoplehood energy data and turn it into device activations')
#parser.add_argument('--participants',type=str, default='participants.csv', help='participants file to read and write')
parser.add_argument('--baseurl',type=str, default='https://sbx-api.mydex.org/api/pds/', help='Base URL for queries')
parser.add_argument('--dataset',type=str, default='ds_energy_consumption', help='Mydex dataset name for energy data')
parser.add_argument('--host',type=str, default='localhost', help='postgres host (localhost is default)')
parser.add_argument('--database',type=str, default='/var/cache/blackwood/energy.db', help='sqlite3 database')
parser.add_argument('--username',type=str, default='peoplehood', help='postgres user')
parser.add_argument('--password',type=str, default='P24pl2h44d', help='postgres password')
parser.add_argument('--dbenergy',type=str, default='energy', help='postgres table for energy data')
parser.add_argument('--dbconnection',type=str, default='connection', help='postgres table for connection data')
parser.add_argument('--apikey',type=str, default='3IbL8UXnkuonNSbTVlEBNekVJLv0jQBX', help='Mydex API Key')
parser.add_argument('--models',type=str, default="./TFLiteModels", help='location of model files')
parser.add_argument('--offset',type=int, default=0, help='offset')
parser.add_argument('--homeid',type=int, help='ID of single home to run process for (leave empty for all homes)')
parser.add_argument('--signal',type=str, default='watts', help='electric mains signal to predict from')
parser.add_argument('--output',type=str, default='predictions', help='output dir for predictions (may not be used)')
parser.add_argument('--pds',type=bool, default=False, help='If True, store activations to PDS')
parser.add_argument('--basedirectory',type=str, default='/home/blackwood/RunLiveNILM', help='Directory for input / output')
parser.add_argument('--activity_rules',type=str, default='./usage-to-use_rules_smart.json', help='rules for turning inferred or actual readings into activations')
parser.add_argument('--activations_to_database',type=bool, default=False, help='send activations to database')
parser.add_argument('--data_rate',type=int, default=10, help='Expected number of seconds between readings')
args = parser.parse_args()

conn = None
cur = None
try:
    conn = sqlite3.connect('/var/cache/blackwood/energy.db')
    cur = conn.cursor()
    print("Database created and Successfully Connected to SQLite")

except sqlite3.Error as error:
    print(error)


# Read the participants file 
#participants = pd.read_csv(args.participants, header=0, sep=',')

# UTC
now = utc.localize(datetime.now())
#now = datetime.utcnow()
startofthishour = datetime.utcnow()
startofthishour = startofthishour.replace(minute=0,second=0,microsecond=0);

lastfetches=()

# Model loading
# The models for kettle and microwave are loaded using the wrapper around the model weights of Cillian's NILM models.
nilm_kettle = NILM_Model('kettle',model_path=args.models)
nilm_microwave = NILM_Model('microwave',model_path=args.models)
nilm_shower = NILM_Model('electricshower',model_path=args.models)
nilm_washing = NILM_Model('washingmachine',model_path=args.models)
nilm_cooker = NILM_Model('electriccooker',model_path=args.models)
nilm_dishwasher = NILM_Model('dishwasher',model_path=args.models)

path = args.basedirectory + "/CSV_" + now.strftime('%Y%m%dT%H%M%S')
actpath = args.basedirectory + "/Activations_" + now.strftime('%Y%m%dT%H%M%S')

try:
    os.mkdir(path)
    os.mkdir(actpath)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)


def write_to_pds(connectionid,uid,key,activation_dataframe,device_name):
    print("write to PDS for ",str(connectionid),"'s",device_name)
    url = args.baseurl + 'transaction.json?uid=' + str(uid) + '&key=' + key + '&api_key=' + args.apikey + '&con_id=' + str(connectionid) + '&source_type=connection&dataset=ds_device_log&instance=0'
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    for index, row in activation_dataframe.iterrows():
        payload = { "ds_device_log": {"action": device_name, "start_time": int(row.start.timestamp()), "end_time": int(row.end.timestamp()), "source": "connection"} }
        print('send',payload,'to',url)
        r = requests.post(url,json=payload,headers=headers)
        print('return',str(r.status_code))

def write_to_db(cursor,disaggrunid,uid,activation_dataframe,device_name):
    print("write to DB for ",str(uid),"'s",device_name)
    
    for index, row in activation_dataframe.iterrows():
        upquery = "UPDATE activation set status='duplicate' where homeid="+str(uid)+" and appliancename='"+device_name+"' and starttime='"+row.start.strftime('%Y-%m-%d %H:%M:%S')+"'"
        container_query = "SELECT * from activation where homeid="+str(uid)+" and appliancename='"+device_name+"' and status='active' and starttime<='"+row.start.strftime('%Y-%m-%d %H:%M:%S')+"' and endtime>='"+row.end.strftime('%Y-%m-%d %H:%M:%S')+"'"
        overlap_query = "SELECT id,starttime,endtime from activation where homeid="+str(uid)+" and appliancename='"+device_name+"' and status='active' and ((starttime<='"+row.start.strftime('%Y-%m-%d %H:%M:%S')+"' and starttime>='"+row.start.strftime('%Y-%m-%d %H:%M:%S')+"') or (endtime<='"+row.end.strftime('%Y-%m-%d %H:%M:%S')+"' and endtime>='"+row.end.strftime('%Y-%m-%d %H:%M:%S')+"'))"
        insstatus='active'
        try:
            print("CONTAINER select: ",container_query)
            cursor.execute(container_query)
            containers = cursor.fetchall()
            if len(containers)>0: # One or more active appliance activations contain the new one - insert the new one as a duplicate
                print("Keep old activation as it has the same or greater extent")
                insstatus='duplicate'
            else:
                print("OVERLAP select: ",overlap_query)
                cursor.execute(overlap_query)                
                new_duration=(row.end-row.start).total_seconds()
                for id,st,et in cursor.fetchall():
                    print("OVERLAP ",str(id))
                    old_duration=(et-st).total_seconds()
                    if (old_duration>new_duration):
                        print("Remove new shorter activation")
                        insstatus='duplicate'
                        break
                    else:
                        upquery = "UPDATE activation set status='duplicate' where id="+str(id)
                        print("Remove old shorter activation",upquery)
                        cursor.execute(upquery)
                
            insquery = "INSERT INTO activation (homeid, disaggregation_run_id, status, fileref, settings, appliancename, starttime, endtime, energy_watts) values ("+str(uid)+","+str(disaggrunid)+",'"+insstatus+"','','','"+device_name+"','"+row.start.strftime('%Y-%m-%d %H:%M:%S')+"','"+row.end.strftime('%Y-%m-%d %H:%M:%S')+"',"+str(row.energy)+")"
            print("INSERT act: ",insquery)
            cursor.execute(insquery)
        except:
            print("FAILED to insert!!")

    conn.commit()


def data_report (runid,uid,stime,etime,df_data,cursor):
    percentage=0
    actual_records=0
    expected_records=-1

    if df_data.empty:        
        print("Activation data for",connid,"is empty! SET DATA ALERT.")
        insstatement = "INSERT INTO DATA_ALERT (runid,userid,alarmtime,description) values ("+str(runid)+","+str(uid)+",'"+now.strftime('%Y-%m-%d %H:%M:%S')+"','No data being reported')"
        print(insstatement)
        try:
            cursor.execute(insstatement)
            conn.commit()
        except:
            print("FAILED to insert data alert!!")

    else:
        try:
            expected_records = math.floor((etime-stime).total_seconds() / args.data_rate)
            # find num seconds between etime and stime and divide by args.data_rate (expected_records)
            actual_records = len(df_data.index)
            if expected_records>0:
                percentage=round((actual_records/expected_records)*100)
        except:
            print("failed to derive percentage")

    try:
        insstatement = "INSERT INTO DATA_REPORT (runid,userid,report_time,start_time,end_time,actual_records,expected_records,percentage) values ("+str(runid)+","+str(uid)+",'"+now.strftime('%Y-%m-%d %H:%M:%S')+"','"+stime.strftime('%Y-%m-%d %H:%M:%S')+"','"+etime.strftime('%Y-%m-%d %H:%M:%S')+"',"+str(actual_records)+","+str(expected_records)+","+str(percentage)+")"
        print(insstatement)
        cursor.execute(insstatement)
        conn.commit()
    except:
        print("FAILED to insert data report!!")


    
def runnilm(connectionid,uid,key,df_data,cursor):
    print("NILM run for "+connectionid)
    print("Data extent "+str(len(df_data.index)))
    ts_predictions_kettle=pd.DataFrame()
    ts_predictions_microwave=pd.DataFrame()
    ts_predictions_shower=pd.DataFrame()
    ts_predictions_washingmc=pd.DataFrame()
    ts_predictions_cooker=pd.DataFrame()
    ts_predictions_dishwasher=pd.DataFrame()
    count=0
    i=args.offset
    while i < (len(df_data.index)-1033):
        df_example = df_data.iloc[i:i+1033,]
        print("Data chunk"+str(df_example[args.signal]))

        # Compute the predictions. The NILM_Model class has a predict() method which takes a pd.Series
        # of length 1033 (the input length of Cillian's model) at 10s sample frequency and returns
        # a pd.Series containing the predicted power. The predictions will only be made for the subset
        # of readings at the center so that each predicted point has the full receptive field available.
        # Predict
        if len(ts_predictions_kettle.index)==0:
            ts_predictions_kettle = nilm_kettle.predict(df_example[args.signal])
            ts_predictions_microwave = nilm_microwave.predict(df_example[args.signal])
            ts_predictions_shower = nilm_shower.predict(df_example[args.signal])
            ts_predictions_washingmc = nilm_washing.predict(df_example[args.signal])
            ts_predictions_cooker = nilm_cooker.predict(df_example[args.signal])
            ts_predictions_dishwasher = nilm_dishwasher.predict(df_example[args.signal])
            #print(ts_predictions_kettle.tail())
        else:
            ts_predictions_kettle = ts_predictions_kettle._append(nilm_kettle.predict(df_example[args.signal]))
            ts_predictions_microwave = ts_predictions_microwave._append(nilm_microwave.predict(df_example[args.signal]))
            ts_predictions_shower = ts_predictions_shower._append(nilm_shower.predict(df_example[args.signal]))
            ts_predictions_washingmc = ts_predictions_washingmc._append(nilm_washing.predict(df_example[args.signal]))
            ts_predictions_cooker = ts_predictions_cooker._append(nilm_cooker.predict(df_example[args.signal]))
            ts_predictions_dishwasher = ts_predictions_dishwasher._append(nilm_dishwasher.predict(df_example[args.signal]))
            #print(ts_predictions_kettle.tail())

        i+=517
        count+=1

    # Save files
    ts_predictions_kettle.to_csv(path+"/home"+str(uid)+"_"+args.signal+"_kettle.csv.gz", header=False, index=True, sep=',', compression='gzip')
    ts_predictions_shower.to_csv(path+"/home"+str(uid)+"_"+args.signal+"_shower.csv.gz", header=False, index=True, sep=',', compression='gzip')
    ts_predictions_microwave.to_csv(path+"/home"+str(uid)+"_"+args.signal+"_microwave.csv.gz", header=False, index=True, sep=',', compression='gzip')
    ts_predictions_washingmc.to_csv(path+"/home"+str(uid)+"_"+args.signal+"_washingmc.csv.gz", header=False, index=True, sep=',', compression='gzip')
    ts_predictions_cooker.to_csv(path+"/home"+str(uid)+"_"+args.signal+"_cooker.csv.gz", header=False, index=True, sep=',', compression='gzip')
    ts_predictions_dishwasher.to_csv(path+"/home"+str(uid)+"_"+args.signal+"_dishwasher.csv.gz", header=False, index=True, sep=',', compression='gzip')


    # Turn into activations
    #showerdetector = ActivationDetector(appliance="electricshower",rulesfile=args.activity_rules,sample_rate=10)
    #shower_activations = showerdetector.get_activations(ts_predictions_shower)
    kettledetector = ActivationDetector(appliance="kettle",rulesfile=args.activity_rules,sample_rate=10)
    kettle_activations = kettledetector.get_activations(ts_predictions_kettle)
    microwavedetector = ActivationDetector(appliance="microwave",rulesfile=args.activity_rules,sample_rate=10)
    microwave_activations = microwavedetector.get_activations(ts_predictions_microwave)
    cookerdetector = ActivationDetector(appliance="electricoven",rulesfile=args.activity_rules,sample_rate=10)
    cooker_activations = cookerdetector.get_activations(ts_predictions_cooker)
    washingmachinedetector = ActivationDetector(appliance="washingmachine",rulesfile=args.activity_rules,sample_rate=10)
    washingmachine_activations = washingmachinedetector.get_activations(ts_predictions_washingmc)
    dishwasherdetector = ActivationDetector(appliance="dishwasher",rulesfile=args.activity_rules,sample_rate=10)
    dishwasher_activations = dishwasherdetector.get_activations(ts_predictions_dishwasher)
    showerdetector = ActivationDetector(appliance="electricshower",rulesfile=args.activity_rules,sample_rate=10)
    shower_activations = showerdetector.get_activations(ts_predictions_shower)

    # record that we ran disaggregation
    firstel=df_data.iloc[0]
    lastel=df_data.iloc[-1]
    print("Run from ",firstel)
    print("Run to ",lastel)
    firsttime=firstel.name
    lasttime=lastel.name
    insquery = "INSERT INTO disaggregation_run (userid,runtime,starttime,endtime) values ('"+str(uid)+"',CURRENT_TIMESTAMP,'"+firsttime.strftime('%Y-%m-%d %H:%M:%S')+"','"+lasttime.strftime('%Y-%m-%d %H:%M:%S')+"')"
    #print("INSERT disagg: ",insquery)
    #try:
    #    cursor.execute(insquery)
    #    conn.commit()
    #except:
    #    print("FAILED to insert!!")
    # Find the ID
    disaggrunid=0;
    #cursor.execute("select last_value from disaggregation_run_runid_seq")
    #disaggrunid = cur.fetchone()[0]
    print("last id is",disaggrunid)
    # Report on data rate
    #data_report(disaggrunid,uid,firsttime,lasttime,df_data,cursor)

    # and save
    try:
        if not shower_activations.empty:
            shower_activations.to_csv(actpath+"/home"+str(uid)+"_"+args.signal+"_shower_activations.csv.gz", header=True, sep=',', compression='gzip')
        if not kettle_activations.empty:
            kettle_activations.to_csv(actpath+"/home"+str(uid)+"_"+args.signal+"_kettle_activations.csv.gz", header=True, sep=',', compression='gzip')
        if not microwave_activations.empty:
            microwave_activations.to_csv(actpath+"/home"+str(uid)+"_"+args.signal+"_microwave_activations.csv.gz", header=True, sep=',', compression='gzip')
        if not cooker_activations.empty:
            cooker_activations.to_csv(actpath+"/home"+str(uid)+"_"+args.signal+"_cooker_activations.csv.gz", header=True, sep=',', compression='gzip')
        if not washingmachine_activations.empty:
            washingmachine_activations.to_csv(actpath+"/home"+str(uid)+"_"+args.signal+"_washingmachine_activations.csv.gz", header=True, sep=',', compression='gzip')

        #if args.activations_to_database:
            #write_to_db(cursor,disaggrunid,uid,kettle_activations,"kettle")
            #write_to_db(cursor,disaggrunid,uid,washingmachine_activations,"washingmachine")
            #write_to_db(cursor,disaggrunid,uid,cooker_activations,"cooker")
            #write_to_db(cursor,disaggrunid,uid,microwave_activations,"microwave")
            #write_to_db(cursor,disaggrunid,uid,shower_activations,"shower")
            #write_to_db(cursor,disaggrunid,uid,dishwasher_activations,"dishwasher")
            #conn.commit()


        #if (args.pds):
        #write_to_pds(connectionid,uid,key,kettle_activations,"kettle")

    except:
        print("ERROR")
        print("ERROR: failed to write disaggregated activations for "+str(uid)+"!")
        print("ERROR")


cads = []
clips = []

# List homes we don't want to disaggregate / apply alert rules to
connectionquery = "SELECT connection_id, uid, mydexidid, key, fields from "+args.dbconnection+" where uid not in (3794,3737,3734,3717,3633,3751,3660)"
#cur.execute(connectionquery)
#conns = cur.fetchall()
joncon = ["3293-40213", 3293, "jonathankil", "Uy1t1ZciuBD3FnRad3HJYLVyf0CXMcqS"]
conns = []
conns.append(joncon)

for row in conns:
    connid=row[0]
    uid=row[1]
    key=row[3]

    if args.homeid and args.homeid!=uid:
        continue

    print("run for "+str(uid))
    
    # Find the most recent alarm run for this participant
    #myquery = "select starttime,endtime from disaggregation_run where userid="+str(uid)+" order by runid desc limit 1"
    #print(myquery)
    #cur.execute(myquery)
    dbt = None
    #dbt = cur.fetchone()
    laststart = None
    lastend= None
    andtimeclause=""
    if dbt is None:
        print("No previous run of disaggregator for home "+str(uid)+". Running for full dataset")
    else:
        laststart = dbt[0]
        lastend = dbt[1]
        print("Previous run from: "+str(laststart)+" to: "+str(lastend))
        lastendtime = lastend + timedelta(hours=-6)
        andtimeclause="and field_energy_reading_timestamp>'"+lastendtime.strftime('%Y-%m-%d %H:%M:%S')+"+00:00'"

    
    energy_query = "select timestamp,watts from "+args.dbenergy+" where source='IDEAL' order by timestamp"
    print(energy_query)
    df_data = pd.read_sql_query(energy_query,conn)
    if df_data.empty:
        print("Energy data for",connid,"is empty")
        # this should note there's a data problem
        #data_report(0,uid,lastend,now,df_data,cur)
    else:
        df_data['timestamp'] = pd.to_datetime(df_data['timestamp'],unit='s')
        df_data.set_index('timestamp',inplace=True)
        # check time of last element - if not greater than endtime of last run, we've got no new data
        lastel=df_data.iloc[-1]
        lasttime=lastel.name
        if dbt is not None and (lastend is not None and lasttime<=(lastend+timedelta(seconds=60))):
            df_data = df_data[0:0]
            print("NULLED OUT DATA")
            data_report(0,uid,lastend,now,df_data,cur)
        else:
            df_data = df_data.resample('10s').mean().ffill(limit=60).fillna(0)
            runnilm(connid,uid,key,df_data,cur)
        #print(df_data.head())

# Update participant file
#participants.to_csv(args.participants,header=True, index=False, sep=',')

if cur is not None:
    cur.close()
if conn is not None:
    conn.close()
