#!/bin/sh
set -e
python $CTRL_ROOT/domain_data/mujoco_worlds/make_xml.py
rm -f $CTRL_DATA/misc/mdp_random_trajs/*
rm -f $CTRL_DATA/misc/mdp_obs_ranges/*
$CTRL_ROOT/build_scripts/generate_random_trajs.py
$CTRL_ROOT/build_scripts/generate_obs_scalings.py
gsutil rsync -R $CTRL_DATA/misc/mdp_obs_ranges/ gs://adp4control/misc/mdp_obs_ranges/
