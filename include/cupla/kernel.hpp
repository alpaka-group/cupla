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

#include "cupla/types.hpp"

#include "cupla/datatypes/dim3.hpp"
#include "cupla/datatypes/uint.hpp"
#include "cupla/manager/Stream.hpp"
#include "cupla/manager/Device.hpp"

#include <utility>


namespace cupla{

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED) || \
    defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED) || \
    defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    /** optimize elemSize and blockSize for a special device
     *
     * This implementation does nothing (empty wrapper)
     */
    struct OptimizeBlockElem
    {
        static void get( dim3 const & , dim3 const &  )
        {

        }
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    /** optimize elemSize and blockSize for a special device
     *
     * This implementation swap elemSize and blockSize
     */
    struct OptimizeBlockElem
    {
        static void get( dim3 & blockSize, dim3 & elemSize )
        {
            std::swap(blockSize,elemSize);
        }
    };
#endif

struct KernelHelper
{
    static cuplaStream_t
    getStream(
        size_t sharedMemSize = 0,
        cuplaStream_t stream = 0
    )
    {
        return stream;
    }

    static size_t
    getSharedMemSize(
        size_t sharedMemSize = 0,
        cuplaStream_t stream = 0
    )
    {
        return sharedMemSize;
    }

};

template<
    typename T_Kernel
>
struct CuplaKernel :
    public T_Kernel
{
    size_t const  m_dynSharedMemBytes;

    CuplaKernel( size_t const & dynSharedMemBytes ) :
        m_dynSharedMemBytes( dynSharedMemBytes )
    { }
};

template<
    typename T_Kernel,
    typename T_Stream,
    typename... T_Args
>
void startKernel(
    T_Kernel const & kernel,
    uint3 const & gridSize,
    uint3 const & blockSize,
    uint3 const & elemPerThread,
    T_Stream & stream,
    T_Args && ... args
){

  auto dev( manager::Device<AccDev>::get().current() );
  ::alpaka::workdiv::WorkDivMembers<
    KernelDim,
    IdxType
  > workDiv(
      static_cast<IdxVec3>(gridSize),
      static_cast<IdxVec3>(blockSize),
      static_cast<IdxVec3>(elemPerThread)
  );
  auto const exec(::alpaka::exec::create<Acc>(workDiv, kernel, args...));
  ::alpaka::stream::enqueue(stream, exec);
}

} // namespace cupla


namespace alpaka
{
namespace kernel
{
namespace traits
{
    //! CuplaKernel has defined the extern shared memory as member
    template<
        typename T_UserKernel,
        typename T_Acc
    >
    struct BlockSharedMemDynSizeBytes<
        ::cupla::CuplaKernel< T_UserKernel >,
        T_Acc
    >
    {
        template<
            typename... TArgs
        >
        ALPAKA_FN_HOST
        static auto
        getBlockSharedMemDynSizeBytes(
            ::cupla::CuplaKernel< T_UserKernel > const & userKernel,
            TArgs const & ...)
        -> ::alpaka::size::Size<T_Acc>
        {
            return userKernel.m_dynSharedMemBytes;
        }
    };
} // namespace traits
} // namespace kernel
} // namespace alpaka



#define CUPLA_CUDA_KERNEL_PARAMS(...)                                          \
    const KernelType theOneAndOnlyKernel( sharedMemSize );                     \
    cupla::startKernel(                                                        \
        theOneAndOnlyKernel,                                                   \
        m_gridSize,                                                            \
        m_blockSize,                                                           \
        m_elemPerThread,                                                       \
        stream,                                                                \
        __VA_ARGS__                                                            \
    );                                                                         \
    }

#define CUPLA_CUDA_KERNEL_CONFIG(gridSize,blockSize,...)                       \
    const cupla::uint3 m_gridSize = dim3(gridSize);                            \
    const cupla::uint3 m_blockSize = dim3(blockSize);                          \
    const cupla::uint3 m_elemPerThread = dim3();                               \
    auto& stream(                                                              \
        cupla::manager::Stream<                                                \
            cupla::AccDev,                                                     \
            cupla::AccStream                                                   \
        >::get().stream(                                                       \
            cupla::KernelHelper::getStream( __VA_ARGS__ )                      \
        )                                                                      \
    );                                                                         \
    size_t const sharedMemSize = cupla::KernelHelper::getSharedMemSize(        \
        __VA_ARGS__                                                            \
    );                                                                         \
    CUPLA_CUDA_KERNEL_PARAMS

/** default cupla kernel call */
#define CUPLA_KERNEL(...) {                                                    \
    using KernelType = ::cupla::CuplaKernel< __VA_ARGS__ >;                    \
    CUPLA_CUDA_KERNEL_CONFIG

#define CUPLA_CUDA_KERNEL_CONFIG_OPTI(gridSize,blockSize,...)                  \
    const cupla::uint3 m_gridSize = dim3(gridSize);                            \
    dim3 tmp_blockSize = dim3( blockSize );                                    \
    dim3 tmp_elemSize;                                                         \
    cupla::OptimizeBlockElem::get( tmp_blockSize, tmp_elemSize );              \
    const cupla::uint3 m_blockSize = tmp_blockSize;                            \
    const cupla::uint3 m_elemPerThread = tmp_elemSize;                         \
    auto& stream(                                                              \
        cupla::manager::Stream<                                                \
            cupla::AccDev,                                                     \
            cupla::AccStream                                                   \
        >::get().stream(                                                       \
            cupla::KernelHelper::getStream( __VA_ARGS__ )                      \
        )                                                                      \
    );                                                                         \
    size_t const sharedMemSize = cupla::KernelHelper::getSharedMemSize(        \
        __VA_ARGS__                                                            \
    );                                                                         \
    CUPLA_CUDA_KERNEL_PARAMS

/** call the kernel with an element layer
 *
 * This kernel call swap the blockSize and the elemSize depending
 * on the activated accelerator.
 */
#define CUPLA_KERNEL_OPTI(...) {                                               \
    using KernelType = ::cupla::CuplaKernel< __VA_ARGS__ >;                    \
    CUPLA_CUDA_KERNEL_CONFIG_OPTI
