#!/bin/bash

set -e
set -o pipefail

# merge the PR to the latest version of the destination branch

cd $CI_PROJECT_DIR

# install all required clang versions
for CXX_COMPILER in $CUPLA_CXX; do
    clang_version=$(echo $CXX_COMPILER | tr -d "clang++-")
    echo "Clang-version: $clang_version"

    if ! agc-manager -e clang@${clang_version}; then
        apt update
        apt install -y clang-${clang_version}
        apt install -y libomp-${clang_version}-dev
    else
        CLANG_BASE_PATH="$(agc-manager -b clang@${clang_version})"
        export PATH=$CLANG_BASE_PATH/bin:$PATH
    fi
done
