#!/bin/bash
# sbatch --clusters=es --partition=dtn_f5_f6 --qos=hpss --nodes=1 --output=log.get_ngfs get_ngfs_data.sh

DATE1=$(date -d "20250814" +%Y%m%d)
DATE2=$(date -d "20250814" +%Y%m%d)

until [[ $DATE1 -gt $DATE2 ]]; do
    
    YYYY=${DATE1:0:4}
    MM=${DATE1:4:2}
    DD=${DATE1:6:2}
    printf -v JD "%03d" "$(date -d "$DATE1" +%j)"
    
    FILE="${DATE1}0000.zip"
    CSV_EAST=NGFS_FIRE_DETECTIONS_GOES-19_ABI_CONUS_${YYYY}_${MM}_${DD}_${JD}.csv
    CSV_WEST=NGFS_FIRE_DETECTIONS_GOES-18_ABI_CONUS_${YYYY}_${MM}_${DD}_${JD}.csv
    
    PATH_EAST=/BMC/fdr/Permanent/${YYYY}/${MM}/${DD}/data/sat/ssec/goes-east/ngfs/${FILE}
    PATH_WEST=/BMC/fdr/Permanent/${YYYY}/${MM}/${DD}/data/sat/ssec/goes-west/ngfs/${FILE}
    
    hsi get ${PATH_EAST}
    unzip ${FILE} ${CSV_EAST} -d data
    rm -f $FILE
    
    hsi get ${PATH_WEST}
    unzip ${FILE} ${CSV_WEST} -d data
    rm -f $FILE
    
    
    DATE1=$(date -d "$DATE1 + 1 day" +%Y%m%d)
done


