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

// emulated that cuda runtime is loaded
#ifndef __DRIVER_TYPES_H__
# define __DRIVER_TYPES_H__
#endif

enum cuplaMemcpyKind
{
  cuplaMemcpyHostToHost,
  cuplaMemcpyHostToDevice,
  cuplaMemcpyDeviceToHost,
  cuplaMemcpyDeviceToDevice
};

enum cuplaError
{
    cuplaSuccess = 0,
    cuplaErrorMemoryAllocation = 2,
    cuplaErrorInitializationError = 3,
    cuplaErrorNotReady = 34
};

enum EventProp
{
    cuplaEventDisableTiming = 2
};

using cuplaError_t = enum cuplaError;


using cuplaStream_t = void*;

using cuplaEvent_t = void*;

