#!/bin/bash

##########################
# update environment
##########################
PATH=$(agc-manager -b cmake@3.22)/bin:$PATH

CUPLA_ROOT=$(pwd)
mkdir build
cd build
cmake .. -Dcupla_BUILD_EXAMPLES=${CUPLA_BUILD_EXAMPLE} -Dcupla_ALPAKA_PROVIDER="external" -DBOOST_ROOT=/opt/boost/${CUPLA_BOOST_VERSION} -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON
cmake --build .
cmake --install .
cd ${CUPLA_ROOT}
