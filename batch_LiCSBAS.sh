#!/bin/bash -eu

# LiCSBAS steps:
#  01: LiCSBAS01_get_geotiff.py
#  02: LiCSBAS02_ml_prep.py
#  03: LiCSBAS03op_GACOS.py (optional)
#  04: LiCSBAS04op_mask_unw.py (optional)
#  05: LiCSBAS05op_clip_unw.py (optional)
#  11: LiCSBAS11_check_unw.py
#  12: LiCSBAS12_loop_closure.py
#  13: LiCSBAS13_sb_inv.py
#  14: LiCSBAS14_vel_std.py
#  15: LiCSBAS15_mask_ts.py
#  16: LiCSBAS16_filt_ts.py

#################
### Settings ####
#################
start_step="01"	# 01-05, 11-16
end_step="16"	# 01-05, 11-16

nlook="1"	# multilook factor, used in step02
GEOCmldir="GEOCml${nlook}"	# If start from 11 or later after doing 03-05, use e.g., GEOCml${nlook}GACOSmaskclip
n_para="" # Number of paralell processing in step 02-05,12,13,16. default: number of usable CPU
check_only="n" # y/n. If y, not run scripts and just show commands to be done

logdir="log"
log="$logdir/$(date +%Y%m%d%H%M)$(basename $0 .sh)_${start_step}_${end_step}.log"

### Optional steps (03-05) ###
do03op_GACOS="n"	# y/n
do04op_mask="n"	# y/n
do05op_clip="n"	# y/n
p04_mask_coh_thre=""	# e.g. 0.2
p04_mask_range=""	# e.g. 10:100/20:200 (ix start from 0)
p04_mask_range_file=""	# Name of file containing range list
p05_clip_range=""	# e.g. 10:100/20:200 (ix start from 0)
p05_clip_range_geo=""	# e.g. 130.11/131.12/34.34/34.6 (in deg)

### Frequently used options. If blank, use default. ###
p11_unw_thre=""	# default: 0.5
p11_coh_thre=""	# default: 0.1
p12_loop_thre=""	# default: 1.5 rad
p15_coh_thre=""	# default: 0.05
p15_n_unw_r_thre=""	# default: 1.5
p15_vstd_thre=""	# default: 100 mm/yr
p15_maxTlen_thre=""	# default: 1 yr
p15_n_gap_thre=""	# default: 10
p15_stc_thre=""	# default: 5 mm
p15_n_ifg_noloop_thre=""	# default: 10
p15_n_loop_err_thre=""	# default: 5
p15_resid_rms_thre=""	# default: 2 mm
p16_filtwidth_km=""	# default: 2 km
p16_filtwidth_yr=""	# default: avg_interval*3 yr
p16_deg_deramp=""	# 1, bl, or 2. default: no deramp
p16_hgt_linear="n"	# y/n. default: n

### Less frequently used options. If blank, use default. ###
p01_frame=""	# e.g. 021D_04972_131213 
p01_start_date=""	# default: 20141001
p01_end_date=""	# default: today
p01_get_gacos="y" # y/n 
p02_GEOCdir=""	# default: GEOC
p02_GEOCmldir=""	# default: GEOCml$nlook
p02_frame=""	# e.g. 021D_04972_131213
p02_n_para=""   # default: # of usable CPU
order_op03_05="03 04 05"	# can change order e.g., 05 03 04
p03_inGEOCmldir=""	# default: $GEOCmldir
p03_outGEOCmldir_suffix="" # default: GACOS
p03_fillhole="y"	# y/n. default: n
p03_gacosdir=""	# default: GACOS
p03_n_para=""   # default: # of usable CPU
p04_inGEOCmldir=""	# default: $GEOCmldir
p04_outGEOCmldir_suffix="" # default: mask
p04_n_para=""   # default: # of usable CPU
p05_inGEOCmldir=""      # default: $GEOCmldir
p05_outGEOCmldir_suffix="" # default: clip
p05_n_para=""   # default: # of usable CPU
p11_GEOCmldir=""	# default: $GEOCmldir
p11_TSdir=""	# default: TS_$GEOCmldir
p12_GEOCmldir=""        # default: $GEOCmldir
p12_TSdir=""    # default: TS_$GEOCmldir
p12_n_para=""	# default: # of usable CPU
p13_GEOCmldir=""        # default: $GEOCmldir
p13_TSdir=""    # default: TS_$GEOCmldir
p13_inv_alg=""	# LS (default) or WLS
p13_mem_size=""	# default: 4000 (MB)
p13_gamma=""	# default: 0.0001
p13_n_para=""	# default: # of usable CPU
p13_n_unw_r_thre=""	# defualt: 1
p13_keep_incfile="n"	# y/n. default: n
p14_TSdir=""    # default: TS_$GEOCmldir
p14_mem_size="" # default: 4000 (MB)
p15_TSdir=""    # default: TS_$GEOCmldir
p15_vmin=""	# default: auto (mm/yr)
p15_vmax=""	# default: auto (mm/yr)
p15_keep_isolated="n"	# y/n. default: n
p15_noautoadjust="n" # y/n. default: n
p16_TSdir=""    # default: TS_$GEOCmldir
p16_hgt_min=""	# default: 200 (m)
p16_hgt_max=""  # default: 10000 (m)
p16_nomask="n"	# y/n. default: n
p16_n_para=""   # default: # of usable CPU


#############################
### Run (No need to edit) ###
#############################
echo ""
echo "Start step: $start_step"
echo "End step:   $end_step"
echo "Log file:   $log"
echo ""
mkdir -p $logdir

if [ $start_step -le 01 -a $end_step -ge 01 ];then
  p01_op=""
  if [ ! -z $p01_frame ];then p01_op="$p01_op -f $p01_frame"; fi
  if [ ! -z $p01_start_date ];then p01_op="$p01_op -s $p01_start_date"; fi
  if [ ! -z $p01_end_date ];then p01_op="$p01_op -e $p01_end_date"; fi
  if [ $p01_get_gacos == "y" ];then p01_op="$p01_op --get_gacos"; fi

  if [ $check_only == "y" ];then
    echo "LiCSBAS01_get_geotiff.py $p01_op"
  else
    LiCSBAS01_get_geotiff.py $p01_op 2>&1 | tee -a $log
    if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
  fi
fi

if [ $start_step -le 02 -a $end_step -ge 02 ];then
  p02_op=""
  if [ ! -z $p02_GEOCdir ];then p02_op="$p02_op -i $p02_GEOCdir";
    else p02_op="$p02_op -i GEOC"; fi
  if [ ! -z $p02_GEOCmldir ];then p02_op="$p02_op -o $p02_GEOCmldir"; fi
  if [ ! -z $nlook ];then p02_op="$p02_op -n $nlook"; fi
  if [ ! -z $p02_frame ];then p02_op="$p02_op -f $p02_frame"; fi
  if [ ! -z $p02_n_para ];then p02_op="$p02_op --n_para $p02_n_para";
  elif [ ! -z $n_para ];then p02_op="$p02_op --n_para $n_para";fi

  if [ $check_only == "y" ];then
    echo "LiCSBAS02_ml_prep.py $p02_op"
  else
    LiCSBAS02_ml_prep.py $p02_op 2>&1 | tee -a $log
    if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
  fi
fi

## Optional steps
for step in $order_op03_05; do ##1

if [ $step -eq 03 -a $start_step -le 03 -a $end_step -ge 03 ];then
  if [ $do03op_GACOS == "y" ]; then
    p03_op=""
    if [ ! -z $p03_inGEOCmldir ];then inGEOCmldir="$p03_inGEOCmldir";
      else inGEOCmldir="$GEOCmldir"; fi
    p03_op="$p03_op -i $inGEOCmldir"
    if [ ! -z $p03_outGEOCmldir_suffix ];then outGEOCmldir="$inGEOCmldir$p03_outGEOCmldir_suffix";
      else outGEOCmldir="${inGEOCmldir}GACOS"; fi
    p03_op="$p03_op -o $outGEOCmldir"
    if [ ! -z $p03_gacosdir ];then p03_op="$p03_op -g $p03_gacosdir"; fi
    if [ $p03_fillhole == "y" ];then p03_op="$p03_op --fillhole"; fi
    if [ ! -z $p03_n_para ];then p03_op="$p03_op --n_para $p03_n_para";
    elif [ ! -z $n_para ];then p03_op="$p03_op --n_para $n_para";fi

    if [ $check_only == "y" ];then
      echo "LiCSBAS03op_GACOS.py $p03_op"
    else
      LiCSBAS03op_GACOS.py $p03_op 2>&1 | tee -a $log
      if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
    fi
    ### Update GEOCmldir to be used for following steps
    GEOCmldir="$outGEOCmldir"
  fi
fi

if [ $step -eq 04 -a $start_step -le 04 -a $end_step -ge 04 ];then
  if [ $do04op_mask == "y" ]; then
    p04_op=""
    if [ ! -z $p04_inGEOCmldir ];then inGEOCmldir="$p04_inGEOCmldir";
      else inGEOCmldir="$GEOCmldir"; fi
    p04_op="$p04_op -i $inGEOCmldir"
    if [ ! -z $p04_outGEOCmldir_suffix ];then outGEOCmldir="$inGEOCmldir$p04_outGEOCmldir_suffix";
      else outGEOCmldir="${inGEOCmldir}mask"; fi
    p04_op="$p04_op -o $outGEOCmldir"
    if [ ! -z $p04_mask_coh_thre ];then p04_op="$p04_op -c $p04_mask_coh_thre"; fi
    if [ ! -z $p04_mask_range ];then p04_op="$p04_op -r $p04_mask_range"; fi
    if [ ! -z $p04_mask_range_file ];then p04_op="$p04_op -f $p04_mask_range_file"; fi
    if [ ! -z $p04_n_para ];then p04_op="$p04_op --n_para $p04_n_para";
    elif [ ! -z $n_para ];then p04_op="$p04_op --n_para $n_para";fi

    if [ $check_only == "y" ];then
      echo "LiCSBAS04op_mask_unw.py $p04_op"
    else
      LiCSBAS04op_mask_unw.py $p04_op 2>&1 | tee -a $log
      if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
    fi
    ### Update GEOCmldir to be used for following steps
    GEOCmldir="$outGEOCmldir"
  fi
fi

if [ $step -eq 05 -a $start_step -le 05 -a $end_step -ge 05 ];then
  if [ $do05op_clip == "y" ]; then
    p05_op=""
    if [ ! -z $p05_inGEOCmldir ];then inGEOCmldir="$p05_inGEOCmldir";
      else inGEOCmldir="$GEOCmldir"; fi
    p05_op="$p05_op -i $inGEOCmldir"
    if [ ! -z $p05_outGEOCmldir_suffix ];then outGEOCmldir="$inGEOCmldir$p05_outGEOCmldir_suffix";
      else outGEOCmldir="${GEOCmldir}clip"; fi
    p05_op="$p05_op -o $outGEOCmldir"
    if [ ! -z $p05_clip_range ];then p05_op="$p05_op -r $p05_clip_range"; fi
    if [ ! -z $p05_clip_range_geo ];then p05_op="$p05_op -g $p05_clip_range_geo"; fi
    if [ ! -z $p05_n_para ];then p05_op="$p05_op --n_para $p05_n_para";
    elif [ ! -z $n_para ];then p05_op="$p05_op --n_para $n_para";fi

    if [ $check_only == "y" ];then
      echo "LiCSBAS05op_clip_unw.py $p05_op"
    else
      LiCSBAS05op_clip_unw.py $p05_op 2>&1 | tee -a $log
      if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
    fi
    ### Update GEOCmldir to be used for following steps
    GEOCmldir="$outGEOCmldir"
  fi
fi

done ##1

### Determine name of TSdir
TSdir="TS_$GEOCmldir"


if [ $start_step -le 11 -a $end_step -ge 11 ];then
  p11_op=""
  if [ ! -z $p11_GEOCmldir ];then p11_op="$p11_op -d $p11_GEOCmldir"; 
    else p11_op="$p11_op -d $GEOCmldir"; fi
  if [ ! -z $p11_TSdir ];then p11_op="$p11_op -t $p11_TSdir"; fi
  if [ ! -z $p11_unw_thre ];then p11_op="$p11_op -u $p11_unw_thre"; fi
  if [ ! -z $p11_coh_thre ];then p11_op="$p11_op -c $p11_coh_thre"; fi

  if [ $check_only == "y" ];then
    echo "LiCSBAS11_check_unw.py $p11_op"
  else
    LiCSBAS11_check_unw.py $p11_op 2>&1 | tee -a $log
    if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
  fi
fi

if [ $start_step -le 12 -a $end_step -ge 12 ];then
  p12_op=""
  if [ ! -z $p12_GEOCmldir ];then p12_op="$p12_op -d $p12_GEOCmldir"; 
    else p12_op="$p12_op -d $GEOCmldir"; fi
  if [ ! -z $p12_TSdir ];then p12_op="$p12_op -t $p12_TSdir"; fi
  if [ ! -z $p12_loop_thre ];then p12_op="$p12_op -l $p12_loop_thre"; fi
  if [ ! -z $p12_n_para ];then p12_op="$p12_op --n_para $p12_n_para";
  elif [ ! -z $n_para ];then p12_op="$p12_op --n_para $n_para";fi

  if [ $check_only == "y" ];then
    echo "LiCSBAS12_loop_closure.py $p12_op"
  else
    LiCSBAS12_loop_closure.py $p12_op 2>&1 | tee -a $log
    if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
  fi
fi

if [ $start_step -le 13 -a $end_step -ge 13 ];then
  p13_op=""
  if [ ! -z $p13_GEOCmldir ];then p13_op="$p13_op -d $p13_GEOCmldir";
    else p13_op="$p13_op -d $GEOCmldir"; fi
  if [ ! -z $p13_TSdir ];then p13_op="$p13_op -t $p13_TSdir"; fi
  if [ ! -z $p13_inv_alg ];then p13_op="$p13_op --inv_alg $p13_inv_alg"; fi
  if [ ! -z $p13_mem_size ];then p13_op="$p13_op --mem_size $p13_mem_size"; fi
  if [ ! -z $p13_gamma ];then p13_op="$p13_op --gamma $p13_gamma"; fi
  if [ ! -z $p13_n_para ];then p13_op="$p13_op --n_para $p13_n_para";
  elif [ ! -z $n_para ];then p13_op="$p13_op --n_para $n_para";fi
  if [ ! -z $p13_n_para ];then p13_op="$p13_op --n_para $p13_n_para"; fi
  if [ ! -z $p13_n_unw_r_thre ];then p13_op="$p13_op --n_unw_r_thre $p13_n_unw_r_thre"; fi
  if [ $p13_keep_incfile == "y" ];then p13_op="$p13_op --keep_incfile"; fi

  if [ $check_only == "y" ];then
    echo "LiCSBAS13_sb_inv.py $p13_op"
  else
    LiCSBAS13_sb_inv.py $p13_op 2>&1 | tee -a $log
    if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
  fi
fi

if [ $start_step -le 14 -a $end_step -ge 14 ];then
  p14_op=""
  if [ ! -z $p14_TSdir ];then p14_op="$p14_op -t $p14_TSdir";
    else p14_op="$p14_op -t $TSdir"; fi
  if [ ! -z $p14_mem_size ];then p14_op="$p14_op --mem_size $p14_mem_size"; fi

  if [ $check_only == "y" ];then
    echo "LiCSBAS14_vel_std.py $p14_op"
  else
    LiCSBAS14_vel_std.py $p14_op 2>&1 | tee -a $log
    if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
  fi
fi

if [ $start_step -le 15 -a $end_step -ge 15 ];then
  p15_op=""
  if [ ! -z $p15_TSdir ];then p15_op="$p15_op -t $p15_TSdir";
    else p15_op="$p15_op -t $TSdir"; fi
  if [ ! -z $p15_coh_thre ];then p15_op="$p15_op -c $p15_coh_thre"; fi
  if [ ! -z $p15_n_unw_r_thre ];then p15_op="$p15_op -u $p15_n_unw_r_thre"; fi
  if [ ! -z $p15_vstd_thre ];then p15_op="$p15_op -v $p15_vstd_thre"; fi
  if [ ! -z $p15_maxTlen_thre ];then p15_op="$p15_op -T $p15_maxTlen_thre"; fi
  if [ ! -z $p15_n_gap_thre ];then p15_op="$p15_op -g $p15_n_gap_thre"; fi
  if [ ! -z $p15_stc_thre ];then p15_op="$p15_op -s $p15_stc_thre"; fi
  if [ ! -z $p15_n_ifg_noloop_thre ];then p15_op="$p15_op -i $p15_n_ifg_noloop_thre"; fi
  if [ ! -z $p15_n_loop_err_thre ];then p15_op="$p15_op -l $p15_n_loop_err_thre"; fi
  if [ ! -z $p15_resid_rms_thre ];then p15_op="$p15_op -r $p15_resid_rms_thre"; fi
  if [ ! -z $p15_vmin ];then p15_op="$p15_op --vmin $p15_vmin"; fi
  if [ ! -z $p15_vmax ];then p15_op="$p15_op --vmax $p15_vmax"; fi
  if [ $p15_keep_isolated == "y" ];then p15_op="$p15_op --keep_isolated"; fi
  if [ $p15_noautoadjust == "y" ];then p15_op="$p15_op --noautoadjust"; fi

  if [ $check_only == "y" ];then
    echo "LiCSBAS15_mask_ts.py $p15_op"
  else
    LiCSBAS15_mask_ts.py $p15_op 2>&1 | tee -a $log
    if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
  fi
fi

if [ $start_step -le 16 -a $end_step -ge 16 ];then
  p16_op=""
  if [ ! -z $p16_TSdir ];then p16_op="$p16_op -t $p16_TSdir";
    else p16_op="$p16_op -t $TSdir"; fi
  if [ ! -z $p16_filtwidth_km ];then p16_op="$p16_op -s $p16_filtwidth_km"; fi
  if [ ! -z $p16_filtwidth_yr ];then p16_op="$p16_op -y $p16_filtwidth_yr"; fi
  if [ ! -z $p16_deg_deramp ];then p16_op="$p16_op -r $p16_deg_deramp"; fi
  if [ $p16_hgt_linear == "y" ];then p16_op="$p16_op --hgt_linear"; fi
  if [ ! -z $p16_hgt_min ];then p16_op="$p16_op --hgt_min $p16_hgt_min"; fi
  if [ ! -z $p16_hgt_max ];then p16_op="$p16_op --hgt_max $p16_hgt_max"; fi
  if [ $p16_nomask == "y" ];then p16_op="$p16_op --nomask"; fi
  if [ ! -z $p16_n_para ];then p16_op="$p16_op --n_para $p16_n_para";
  elif [ ! -z $n_para ];then p16_op="$p16_op --n_para $n_para";fi

  if [ $check_only == "y" ];then
    echo "LiCSBAS16_filt_ts.py $p16_op"
  else
    LiCSBAS16_filt_ts.py $p16_op 2>&1 | tee -a $log
    if [ ${PIPESTATUS[0]} -ne 0 ];then exit 1; fi
  fi
fi

if [ $check_only == "y" ];then
  echo ""
  echo "Above commands will run when you change check_only to \"n\""
  echo ""
fi

