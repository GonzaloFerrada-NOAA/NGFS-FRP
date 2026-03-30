#!/bin/bash
unalias date

# User defined:

DATE1=$( date -ud "2025-08-15 00:00:00" +%s )
DATE2=$( date -ud "2025-08-25 23:00:00" +%s )
DT_MIN=60 # min

PATH_WEST="/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/FIRE/NGFS/data"
PATH_EAST="/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/FIRE/NGFS/data"
PYTHON_ENV="/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/miniconda3/bin/python"

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# No further modifications needed after this line
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
DT=$((${DT_MIN}*60))  # in seconds

for (( t=DATE1; t<=DATE2; t+=DT )); do
    
    CDATE=$(date -ud "@$t" "+%Y-%m-%d_%H:%M:%S")
    echo "$CDATE"
    
    ${PYTHON_ENV} process_bysat_NGFS.py "${CDATE}" ${DT_MIN} "${PATH_WEST}" "${PATH_EAST}"
    
done



