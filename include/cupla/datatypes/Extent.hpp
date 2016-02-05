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

namespace cupla
{

    struct Extent{
        MemSizeType width, height, depth;

        ALPAKA_FN_HOST_ACC
        Extent() = default;

        ALPAKA_FN_HOST_ACC
        Extent(
            MemSizeType const w,
            MemSizeType const h,
            MemSizeType const d
        ) :
            width( w ),
            height( h ),
            depth( d )
        {}

        template<
          typename TDim,
          typename TSize,
          typename = typename std::enable_if<
              (TDim::value == 3u)
          >::type
        >
        ALPAKA_FN_HOST_ACC
        Extent(
            ::alpaka::Vec<
                TDim,
                TSize
            > const &vec
        )
        {
            for( uint32_t i(0); i < 3u; ++i ) {
                // alpaka vectors are z,y,x.
                ( &this->width )[ i ] = vec[ ( 3u - 1u ) - i ];
            }
        }

        ALPAKA_FN_HOST_ACC
        operator ::alpaka::Vec<
            cupla::AlpakaDim< 3u >,
            MemSizeType
        >(void) const
        {
            ::alpaka::Vec<
                cupla::AlpakaDim< 3u >,
                MemSizeType
            > vec( depth, height, width );
            return vec;
        }
    };

} //namespace cupla


namespace alpaka
{
namespace dim
{
namespace traits
{

    //! dimension get trait specialization
    template<>
    struct DimType<
        cupla::Extent
    >{
      using type = ::alpaka::dim::DimInt<3u>;
    };

} // namespace traits
} // namespace dim

namespace elem
{
namespace traits
{

    //! element type trait specialization
    template<>
    struct ElemType<
        cupla::Extent
    >{
        using type = cupla::MemSizeType;
    };

} // namespace traits
} // namspace elem

namespace extent
{
namespace traits
{

    //! extent get trait specialization
    template<
        typename T_Idx
    >
    struct GetExtent<
        T_Idx,
        cupla::Extent,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{

        ALPAKA_FN_HOST_ACC
        static auto
        getExtent( cupla::Extent const & extents )
        -> cupla::MemSizeType {
        return (&extents.width)[(3u - 1u) - T_Idx::value];
      }
    };

    //! extent set trait specialization
    template<
        typename T_Idx,
        typename T_Extent
    >
    struct SetExtent<
        T_Idx,
        cupla::Extent,
        T_Extent,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        setExtent(
            cupla::Extent &extents,
            T_Extent const &extent
        )
        -> void
        {
            (&extents.width)[(3u - 1u) - T_Idx::value] = extent;
        }
    };
} // namespace traits
} // namespace extent

namespace offset
{
namespace traits
{

    //! offset get trait specialization
    template<
        typename T_Idx
    >
    struct GetOffset<
        T_Idx,
        cupla::Extent,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        getOffset( cupla::Extent const & offsets )
        -> cupla::MemSizeType{
            return (&offsets.width)[(3u - 1u) - T_Idx::value];
        }
    };


    //! offset set trait specialization.
    template<
        typename T_Idx,
        typename T_Offset
    >
    struct SetOffset<
        T_Idx,
        cupla::Extent,
        T_Offset,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        setOffset(
            cupla::Extent &offsets,
            T_Offset const &offset
        )
        -> void {
            offsets[(3u - 1u) - T_Idx::value] = offset;
        }
    };
} // namespace traits
} // namespace offset

namespace size
{
namespace traits
{

    //! size type trait specialization.
    template<>
    struct SizeType<
        cupla::Extent
    >{
        using type = cupla::MemSizeType;
    };

} // namespace traits
} // namespace size
} // namespave alpaka
