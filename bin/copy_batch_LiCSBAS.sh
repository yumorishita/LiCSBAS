#!/bin/bash
echo ""
echo "Copying batch_LiCSBAS.sh to current directory by:"
echo "cp -i $LICSBAS_PATH/bin/batch_LiCSBAS.sh ."
cp -i $LICSBAS_PATH/bin/batch_LiCSBAS.sh .
chmod 755 batch_LiCSBAS.sh

echo ""
echo "Edit batch_LiCSBAS.sh, then run ./batch_LiCSBAS.sh"
echo
