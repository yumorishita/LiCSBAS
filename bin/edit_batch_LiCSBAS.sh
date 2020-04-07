#!/bin/bash -eu
#
# Edit parameters in batch_LiCSBAS.sh.
#
# v1.0 20200407 Yu Morishita, GSI
#
ver="1.0";day="20200407";auther="Y. Morishita, GSI"

if [ $# -lt 2 ];then
  echo ""
  echo "$(basename $0) ver$ver $day $auther"
  echo "Edit parameters in batch_LiCSBAS.sh"
  echo ""
  echo "Usage: $(basename $0) batch_LiCSBAS param_list"
  echo "  batch_LiCSBAS : Batch script (batch_LiCSBAS.sh) to be edited (overwritten)."
  echo "  param_list : List of parameters and values in batch_LiCSBAS."
  echo "     (Format: param value (e.g., start_step 11))"
  echo ""
  exit 1
fi

### Check files
if [ ! -f $1 ];then
  echo "No $1 exists!"
  echo ""
  exit 1
else
  batch_LiCSBAS="$1"
fi

if [ ! -f $2 ];then
  echo "No $2 exists!"
  echo ""
  exit 1
else
  paramfile="$2"
fi


### Edit parameters
echo ""
while read line
do
  param=$(echo $line | awk '{print $1}')
  value=$(echo $line | awk '{print $2}')

  ### Check if param exist
  num=$(grep -c "^${param}=\".*\"" $batch_LiCSBAS)
  if [ $num -eq 0 ];then
    echo "No $param exists in $batch_LiCSBAS, skip editing."
  else
    value_old=$(grep "^${param}=\".*\"" $batch_LiCSBAS | head -1 | cut -f2 -d\")
    echo "Edit $param from \"$value_old\" to \"$value\""
    sed -i -e "s#${param}=\"${value_old}\"#${param}=\"${value}\"#" $batch_LiCSBAS
  fi

done < $paramfile

echo ""
echo "Finished! Check $batch_LiCSBAS."
echo ""
