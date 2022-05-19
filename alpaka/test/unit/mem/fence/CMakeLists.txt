#
# Copyright 2021 Jan Stephan
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

if(NOT alpaka_ACC_ANY_BT_OACC_ENABLE) # Fences are not supported for OpenACC
        set(_TARGET_NAME "FenceTest")

        append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

        alpaka_add_executable(
                ${_TARGET_NAME}
                ${_FILES_SOURCE})

        target_link_libraries(
                ${_TARGET_NAME}
                PRIVATE common)

        set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/unit")

        add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_alpaka_TEST_OPTIONS})
endif()