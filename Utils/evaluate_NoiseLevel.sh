#!/bin/sh

EPOCHFILEPATH="/work/huchida/SpeakerVerification/SparseSincnet/exp/SincNet_TIMIT/Kwinners/cnn_556-dnn1_777/4_epoch1048/checkpoint/*"
EPOCHFILENAME=${EPOCHFILEPATH##*/}

EPOCHFILENAMES=`find $EPOCHFILEPATH -maxdepth 0 -type f -name *.pkl`

# count=1
for filename in $EPOCHFILENAMES;
do
    echo $filename
    pt_file=`sed -n 7p /work/huchida/SpeakerVerification/SparseSincnet/cfg/evaluate/SincNet_TIMIT.cfg`;
    sed -i '7d' /work/huchida/SpeakerVerification/SparseSincnet/cfg/evaluate/SincNet_TIMIT.cfg
    sed -i "7i pt_file=${filename}" /work/huchida/SpeakerVerification/SparseSincnet/cfg/evaluate/SincNet_TIMIT.cfg
    #echo ${pt_file}
    python /work/huchida/SpeakerVerification/SparseSincnet/speaker_id_evaluate_bgn.py --cfg=cfg/evaluate/SincNet_TIMIT.cfg
    # count=`expr $count + 1`
done
