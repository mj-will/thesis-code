#!/usr/bin/env bash
delay=$1
dims=$2
npool=$3
sed -i "/delay = */c\delay = $delay" run_scaling.submit
sed -i "/dims = */c\dims = $dims" run_scaling.submit
sed -i "/npool = */c\npool = $npool" run_scaling.submit
condor_submit run_scaling.submit
