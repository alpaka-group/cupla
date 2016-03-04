/**
 * Copyright 2015-2016 Rene Widera, Maximilian Knespel
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

#define __syncthreads(...) ::alpaka::block::sync::syncBlockThreads(acc)

#define cudaSuccess cuplaSuccess
#define cudaErrorMemoryAllocation cuplaErrorMemoryAllocation
#define cudaErrorInitializationError cuplaErrorInitializationError
#define cudaErrorNotReady cuplaErrorNotReady

#define cudaError_t cuplaError_t
#define cudaError cuplaError

#define cudaEvent_t cuplaEvent_t

#define cudaStream_t cuplaStream_t

#define dim3 cupla::dim3
#define cudaExtent cupla::Extent
#define cudaPitchedPtr cupla::PitchedPtr

#ifdef cudaEventDisableTiming
#undef cudaEventDisableTiming
#endif
/* cudaEventDisableTiming is a define in CUDA therefore we must remove
 * the old definition with the cupla enum
 */
#define cudaEventDisableTiming cuplaEventDisableTiming

#define sharedMem(ppName, ...)                                                 \
  __VA_ARGS__ &ppName =                                                        \
      ::alpaka::block::shared::st::allocVar<__VA_ARGS__, __COUNTER__>(acc)

#define sharedMemExtern(ppName, ...)                                           \
    __VA_ARGS__ *ppName =                                                      \
        ::alpaka::block::shared::dyn::getMem<__VA_ARGS__>(acc)

#define cudaMemcpyKind cuplaMemcpyKind
#define cudaMemcpyHostToDevice cuplaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost cuplaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice cuplaMemcpyDeviceToDevice
#define cudaMemcpyHostToHost cuplaMemcpyHostToHost

// index renaming
#define blockIdx                                                               \
  static_cast<cupla::uint3>(                                                \
      ::alpaka::idx::getIdx<::alpaka::Grid, ::alpaka::Blocks>(acc))
#define threadIdx                                                              \
  static_cast<cupla::uint3>(                                                \
      ::alpaka::idx::getIdx<::alpaka::Block, ::alpaka::Threads>(acc))

#define gridDim                                                                \
  static_cast<cupla::uint3>(                                                \
      ::alpaka::workdiv::getWorkDiv<::alpaka::Grid, ::alpaka::Blocks>(acc))
#define blockDim                                                               \
  static_cast<cupla::uint3>(                                                \
      ::alpaka::workdiv::getWorkDiv<::alpaka::Block, ::alpaka::Threads>(acc))
#define elemDim                                                               \
  static_cast<cupla::uint3>(                                                \
      ::alpaka::workdiv::getWorkDiv<::alpaka::Thread, ::alpaka::Elems>(acc))

// atomic functions
#define atomicAdd(ppPointer,ppValue) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Add>(acc, ppPointer, ppValue)
#define atomicSub(ppPointer,ppValue) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Sub>(acc, ppPointer, ppValue)
#define atomicMin(ppPointer,ppValue) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Min>(acc, ppPointer, ppValue)
#define atomicMax(ppPointer,ppValue) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Max>(acc, ppPointer, ppValue)
#define atomicInc(ppPointer,ppValue) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Inc>(acc, ppPointer, ppValue)
#define atomicDec(ppPointer,ppValue) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Dec>(acc, ppPointer, ppValue)
#define atomicExch(ppPointer,ppValue) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Exch>(acc, ppPointer, ppValue)
#define atomicCAS(ppPointer,ppCompare,ppValue) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Cas>(acc, ppPointer, ppCompare, ppValue)

// recast functions
/* defining these as inling functions will result in multiple declaration
 * errors when using a CUDA accelerator with alpaka.
 * Note that there may be no spaces between the macro function name and
 * the argument parentheses. */
namespace cupla {


    /* no matter how ALPAKA_FN_HOST_ACC is defined, we want to inline this */
    template< class A, class B >
    ALPAKA_FN_NO_INLINE_HOST_ACC inline
    B __A_as_B( A const & x )
    {
        static_assert( sizeof(A) == sizeof(B), "reinterpretation assumes data types of same size!" );
        union ba { B b; A a; };
        ba tmp;
        tmp.a = x;
        return tmp.b;
    }


} // namespace cupla

#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
#   define __int_as_float( cupla::__A_as_B< int, float > )
#   define __float_as_int( cupla::__A_as_B< float, int > )
#   define __longlong_as_double( cupla::__A_as_B< long long, double > )
#   define __double_as_longlong( cupla::__A_as_B< double, long long > )
#endif
