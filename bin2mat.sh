#!/bin/bash
IWM=$(pwd)
DATADIR=$(pwd)
DATADIR=database/www.physionet.org/physiobank/database/ptbdb/patient*
DBDIR=$(pwd)/mat_data
mkdir mat_data
for FILE in $DATADIR
do
	echo "##################"
	echo $FILE
	
	NUM=$(echo $FILE | egrep -o [0-9]+)
	echo $NUM
	#echo $FILE | egrep -o [0-9]+
	TEMPDIR=$DBDIR/patient$NUM
	echo $TEMPDIR
	mkdir $TEMPDIR
	for FILE2 in $FILE/*dat
	do
		filename="${FILE2##*/}"
		filename="${filename%.*}"
		echo $FILE
		echo $filename
		wfdb2mat -r $FILE/$filename > $filename.info
		mv $IWM/"$filename"m.hea $IWM/mat_data/patient$NUM/"$filename"m.hea
		mv $IWM/"$filename"m.mat $IWM/mat_data/patient$NUM/"$filename"m.mat
		mv $IWM/$filename.info $IWM/mat_data/patient$NUM/$filename.info
		#rm $filename.hea $filename.mat $filename.info
	done
done
echo $cwd
