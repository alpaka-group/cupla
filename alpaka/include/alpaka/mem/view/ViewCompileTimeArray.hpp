/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/PltfCpu.hpp>

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for fixed idx arrays.
    //
    // This allows the usage of multidimensional compile time arrays e.g. int[4][3] as argument to memory ops.
    /*namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed idx array device type trait specialization.
            template<
                typename TFixedSizeArray>
            struct DevType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = dev::DevCpu;
            };

            //#############################################################################
            //! The fixed idx array device get trait specialization.
            template<
                typename TFixedSizeArray>
            struct GetDev<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    TFixedSizeArray const & view)
                -> dev::DevCpu
                {
                    // \FIXME: CUDA device?
                    return pltf::getDevByIdx<pltf::PltfCpu>(0u);
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed idx array dimension getter trait specialization.
            template<
                typename TFixedSizeArray>
            struct DimType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = dim::DimInt<std::rank<TFixedSizeArray>::value>;
            };
        }
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed idx array memory element type get trait specialization.
            template<
                typename TFixedSizeArray>
            struct ElemType<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = typename std::remove_all_extent<TFixedSizeArray>::type;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed idx array width get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TFixedSizeArray>
            struct GetExtent<
                TIdxIntegralConst,
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::rank<TFixedSizeArray>::value > TIdxIntegralConst::value)
                    && (std::extent<TFixedSizeArray, TIdxIntegralConst::value>::value > 0u)>::type>
            {
                //-----------------------------------------------------------------------------
                static constexpr auto getExtent(
                    TFixedSizeArray const & //extent
                )
                -> idx::Idx<TFixedSizeArray>
                {
                    // C++14 constexpr with void return
                    //alpaka::ignore_unused(extent);
                    return std::extent<TFixedSizeArray, TIdxIntegralConst::value>::value;
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The fixed idx array native pointer get trait specialization.
                template<
                    typename TFixedSizeArray>
                struct GetPtrNative<
                    TFixedSizeArray,
                    typename std::enable_if<
                        std::is_array<TFixedSizeArray>::value>::type>
                {
                    using TElem = typename std::remove_all_extent<TFixedSizeArray>::type;

                    //-----------------------------------------------------------------------------
                    static auto getPtrNative(
                        TFixedSizeArray const & view)
                    -> TElem const *
                    {
                        return view;
                    }
                    //-----------------------------------------------------------------------------
                    static auto getPtrNative(
                        TFixedSizeArray & view)
                    -> TElem *
                    {
                        return view;
                    }
                };

                //#############################################################################
                //! The fixed idx array pitch get trait specialization.
                template<
                    typename TFixedSizeArray>
                struct GetPitchBytes<
                    dim::DimInt<std::rank<TFixedSizeArray>::value - 1u>,
                    TFixedSizeArray,
                    typename std::enable_if<
                        std::is_array<TFixedSizeArray>::value
                        && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value > 0u)>::type>
                {
                    using TElem = typename std::remove_all_extent<TFixedSizeArray>::type;

                    //-----------------------------------------------------------------------------
                    static constexpr auto getPitchBytes(
                        TFixedSizeArray const &)
                    -> idx::Idx<TFixedSizeArray>
                    {
                        return sizeof(TElem) * std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value;
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed idx array offset get trait specialization.
            template<
                typename TIdx,
                typename TFixedSizeArray>
            struct GetOffset<
                TIdx,
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                //-----------------------------------------------------------------------------
                static auto getOffset(
                    TFixedSizeArray const &)
                -> idx::Idx<TFixedSizeArray>
                {
                    return 0u;
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The std::vector idx type trait specialization.
            template<
                typename TFixedSizeArray>
            struct IdxType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = std::size_t;
            };
        }
    }*/
}
