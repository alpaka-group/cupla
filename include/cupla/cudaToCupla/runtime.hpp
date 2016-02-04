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

#define cudaMalloc(...) cuplaMalloc(__VA_ARGS__)

#define cudaGetErrorString(...) cuplaGetErrorString(__VA_ARGS__)

#define cudaFree(...) cuplaFree(__VA_ARGS__)

#define cudaSetDevice(...) cuplaSetDevice(__VA_ARGS__)

#define cudaGetDevice(...) cuplaGetDevice(__VA_ARGS__)

#define cudaEventCreate(...) cuplaEventCreate(__VA_ARGS__)
#define cudaEventDestroy(...) cuplaEventDestroy(__VA_ARGS__)

#define cudaEventRecord(...) cuplaEventRecord(__VA_ARGS__)

#define cudaEventElapsedTime(...) cuplaEventElapsedTime(__VA_ARGS__)

#define cudaEventSynchronize(...) cuplaEventSynchronize(__VA_ARGS__)

#define cudaMemcpy(...) cuplaMemcpy(__VA_ARGS__)

#define cudaDeviceReset() cuplaDeviceReset()

#define cudaDeviceSynchronize() cuplaDeviceSynchronize()
