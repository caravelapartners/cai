#!/bin/bash

input_dir="/mnt/disks/era5land/neural/"
output_dir="/mnt/disks/era5land/neural/t/850/"


for file in $input_dir/forecast_*.nc; do
    # Loop over each unique model date and hour (adjust the date/hour pattern accordingly)
    filename=$(basename "$file")
    cdo -f nc sellevel,850 -selname,temperature $file $output_dir${filename::-2}nc
done


