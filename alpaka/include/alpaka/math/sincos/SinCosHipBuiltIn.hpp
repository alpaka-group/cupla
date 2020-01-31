/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/math/sincos/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
    #include <cuda_runtime_api.h>
#else
    #if BOOST_COMP_HCC || BOOST_COMP_HIP
        #include <math_functions.h>
    #else
        #include <math_functions.hpp>
    #endif
#endif

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! sincos.
        class SinCosHipBuiltIn : public concepts::Implements<ConceptMathSinCos, SinCosHipBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! sincos trait specialization.
            template<>
            struct SinCos<SinCosHipBuiltIn, double>
            {
                __device__ static auto sincos(
                    SinCosHipBuiltIn const & sincos_ctx,
                    double const & arg,
                    double & result_sin,
                    double & result_cos)
                -> void
                {
                    alpaka::ignore_unused(sincos_ctx);
                    ::sincos(arg, &result_sin, &result_cos);
                }
            };

            //! The sincos float specialization.
            template<>
            struct SinCos<SinCosHipBuiltIn, float>
            {
                __device__ static auto sincos(
                    SinCosHipBuiltIn const & sincos_ctx,
                    float const & arg,
                    float & result_sin,
                    float & result_cos)
                -> void
                {
                    alpaka::ignore_unused(sincos_ctx);
                    ::sincosf(arg, &result_sin, &result_cos);
                }
            };
        }
    }
}

#endif
