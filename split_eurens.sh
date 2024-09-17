#!/bin/bash

input_dir="/mnt/disks/era5land/eurens/raw"
output_dir="/mnt/disks/era5land/eurens/"


echo "write \"$output_dir/[shortName]/[level]/[date][time].grib\";" > split.filter


for file in $input_dir/*.grib; do
  # Loop over each unique model date and hour (adjust the date/hour pattern accordingly)
    grib_filter split.filter $file
    mv $file $file.done
done

rm split.filter;
