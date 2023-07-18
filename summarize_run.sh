has_p_option=false
while getopts r:p flag
do
  case "$flag" in
    r) run=${OPTARG};;
    p) has_p_option=true;;
    :) echo "Missing argument for option -$OPTARG"; exit 1;;
    \?) echo "Unknown argument -$OPTARG"; exit 1;;
  esac
done

if $has_p_option; then
  /data/software/anaconda2/envs/sctcamsoft/bin/python /home/ctauser/CameraSoftware/trunk/analysis/psct_summarize_run.py -r "$run" -p
else
  if [ -e /data/local_outputDir/cal"$run".r1 ]
  then
    echo "Calibrated data file for run $run exists"
  else
    /data/software/anaconda2/envs/sctcamsoft/bin/apply_calibration_SCT -i /data/local_outputDir/run"$run".fits -p /data/local_outputDir/ped328587.tcal -o /data/local_outputDir/cal"$run".r1
  fi
  /data/software/anaconda2/envs/sctcamsoft/bin/python /home/ctauser/CameraSoftware/trunk/analysis/psct_summarize_run.py -r "$run"
fi
