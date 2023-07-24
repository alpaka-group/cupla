#!/bin/bash

##########################
# update environment
##########################
PATH=$(agc-manager -b cmake@3.22)/bin:$PATH

CUPLA_ROOT=$(pwd)
cd alpaka
mkdir build
cd build
cmake .. -DBOOST_ROOT=/opt/boost/${CUPLA_BOOST_VERSION}
cmake --build .
cmake --install .
cd ${CUPLA_ROOT}
# Remove alpaka repository so that it cannot be used accidentally via add_subdirectory()
rm -rf alpaka
