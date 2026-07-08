#!/bin/sh
hours_runtime = "00"
minutes_runtime = "30"
seconds_runtime = "00"

group = "tga-lbmcity"


iqrsh -g $group -l h_rt=$hours_runtime:$minutes_runtime:$seconds_runtime