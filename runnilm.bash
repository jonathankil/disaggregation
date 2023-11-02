#!/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin://group/ideal/Peoplehood/NILM/RunLiveNILM

source /home/ideal/anaconda2/bin/activate
echo "running NILM"
d=$(date +"%y-%m-%d")
date
echo $d
# source /home/ideal/anaconda2/envs/Smile/lib/python3.8/venv/scripts/common/activate
#source activate Smile
python --version
cd /home/blackwood/RunLiveNILM
python LiteNILMprocess.py > nilmlog_$d\.txt
