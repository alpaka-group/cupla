/**
 * Copyright 2016 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include <alpaka/alpaka.hpp>
#include <cstdint>

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
#   undef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
#   define ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED 1
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#   undef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#   define ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED 1
#endif

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#   undef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#   define ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED 1
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#   undef ALPAKA_ACC_GPU_CUDA_ENABLED
#   define ALPAKA_ACC_GPU_CUDA_ENABLED 1
#endif

#define CUPLA_NUM_SELECTED_DEVICES (                                           \
        ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED +                                  \
        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED +                               \
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED  +                                 \
        ALPAKA_ACC_GPU_CUDA_ENABLED                                            \
    )

#if( CUPLA_NUM_SELECTED_DEVICES > 1 )
    #error "please select only one accelerator"
#endif

#if( CUPLA_NUM_SELECTED_DEVICES == 0 )
    #error "there is no accelerator selected, please run `ccmake .` and select one"
#endif

namespace cupla {


    using MemSizeType = size_t;
    using IdxType = unsigned int;

    static constexpr uint32_t Dimensions = 3u;

    template<
        uint32_t T_dim
    >
    using AlpakaDim = ::alpaka::dim::DimInt< T_dim >;

    using KernelDim = AlpakaDim< Dimensions >;

    using IdxVec3 = ::alpaka::Vec<
        KernelDim,
        IdxType
    >;

    template<
        uint32_t T_dim
    >
    using MemVec = ::alpaka::Vec<
        AlpakaDim< T_dim >,
        MemSizeType
    >;

    using AccHost = ::alpaka::dev::DevCpu;

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED) ||                            \
    defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED) ||                         \
    defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)

    using AccDev = ::alpaka::dev::DevCpu;
    using AccStream = ::alpaka::stream::StreamCpuAsync;

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    using Acc = ::alpaka::acc::AccCpuOmp2Threads<
        KernelDim,
        IdxType
    >;
#endif

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    using Acc = ::alpaka::acc::AccCpuOmp2Blocks<
        KernelDim,
        IdxType
    >;
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    using Acc = ::alpaka::acc::AccCpuThreads<
        KernelDim,
        IdxType
    >;
#endif

#endif


#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using AccDev = ::alpaka::dev::DevCudaRt;
    using AccStream = ::alpaka::stream::StreamCudaRtAsync;
    using Acc = ::alpaka::acc::AccGpuCudaRt<
        KernelDim,
        IdxType
    >;
#endif

    template<
        uint32_t T_dim
    >
    using AccBuf = ::alpaka::mem::buf::Buf<
        AccDev,
        uint8_t,
        AlpakaDim< T_dim >,
        MemSizeType
    >;

    template<
        uint32_t T_dim
    >
    using HostBuf = ::alpaka::mem::buf::Buf<
        AccHost,
        uint8_t,
        AlpakaDim< T_dim >,
        MemSizeType
    >;

    template<
        unsigned T_dim
    >
    using HostBufWrapper =
        ::alpaka::mem::view::ViewPlainPtr<
            AccHost,
            uint8_t,
            AlpakaDim< T_dim >,
            MemSizeType
        >;

    template<
        unsigned T_dim
    >
    using DeviceBufWrapper =
        ::alpaka::mem::view::ViewPlainPtr<
            AccDev,
            uint8_t,
            AlpakaDim< T_dim >,
            MemSizeType
        >;
} // namepsace cupla

