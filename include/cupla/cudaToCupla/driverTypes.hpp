/**
 * Copyright 2015-2016 Rene Widera
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

#define cudaEvent_t cuplaEvent_t

#define cudaStream_t cuplaStream_t

#define dim3 cupla::dim3

#define sharedMem(ppName, ...)                                                 \
  __VA_ARGS__ &ppName =                                                        \
      ::alpaka::block::shared::st::allocVar<__VA_ARGS__, __COUNTER__>(acc)


#define cudaMemcpyHostToDevice cuplaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost cuplaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice cuplaMemcpyDeviceToDevice

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
