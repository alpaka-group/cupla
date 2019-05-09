/* Copyright 2016 Rene Widera
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

#include "cupla/namespace.hpp"
#include "cupla/types.hpp"
#include "cupla_driver_types.hpp"
#include "cupla_runtime.hpp"
#include "cupla/manager/Device.hpp"
#include "cupla/manager/Event.hpp"
#include "cupla/manager/Memory.hpp"
#include "cupla/manager/Stream.hpp"

inline namespace CUPLA_ACCELERATOR_NAMESPACE
{

inline const char *
cuplaGetErrorString(cuplaError_t e)
{
    return CuplaErrorCode::message_cstr(e);
}


/** returns the last error from a runtime call.
 *
 * This call reset the error code to cuplaSuccess
 * @warning If a non CUDA Alpaka backend is used this function will return always cuplaSuccess
 *
 * @return cuplaSuccess if there was no error else the corresponding error type
 */
inline cuplaError_t
cuplaGetLastError()
{
#if( ALPAKA_ACC_GPU_CUDA_ENABLED == 1 )
    // reset the last cuda error
    return (cuplaError_t)cudaGetLastError();
#elif( ALPAKA_ACC_GPU_HIP_ENABLED == 1 )
    return (cuplaError_t)hipGetLastError();
#else
    return cuplaSuccess;
#endif
}


/** returns the last error from a runtime call.
 *
 * This call does not reset the error code.
 * @warning If a non CUDA Alpaka backend is used this function will return always cuplaSuccess
 *
 * @return cuplaSuccess if there was no error else the corresponding error type
 */
inline cuplaError_t
cuplaPeekAtLastError()
{
#if( ALPAKA_ACC_GPU_CUDA_ENABLED == 1 )
    return (cuplaError_t)cudaPeekAtLastError();
#elif( ALPAKA_ACC_GPU_HIP_ENABLED == 1 )
    return (cuplaError_t)hipPeekAtLastError();
#else
    return cuplaSuccess;
#endif
}

} // namespace CUPLA_ACCELERATOR_NAMESPACE
