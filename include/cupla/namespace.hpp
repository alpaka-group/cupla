/* Copyright 2019 Andrea Bocci
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

#if CUPLA_STREAM_ASYNC_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_seq_omp2_async
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_seq_threads_async
#endif

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_omp2_seq_async
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_cuda_async
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_seq_seq_async
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_tbb_seq_async
#endif

#else // CUPLA_STREAM_ASYNC_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_seq_omp2_sync
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_seq_threads_sync
#endif

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_omp2_seq_sync
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_cuda_sync
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_seq_seq_sync
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#define CUPLA_ACCELERATOR_NAMESPACE cupla_tbb_seq_sync
#endif

#endif // CUPLA_STREAM_ASYNC_ENABLED

/*
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
} //namespace CUPLA_ACCELERATOR_NAMESPACE
*/
