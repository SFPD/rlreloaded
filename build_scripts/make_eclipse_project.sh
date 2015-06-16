#!/bin/sh
### Set BUILD_DIR and SOURCE_DIR appropriately
BUILD_DIR=~/build
SOURCE_DIR=~/Proj/control/cpp
set -e
cd $BUILD_DIR
rm -rf control-eclipse
mkdir control-eclipse
cd control-eclipse
cmake -G"Eclipse CDT4 - Unix Makefiles" $SOURCE_DIR