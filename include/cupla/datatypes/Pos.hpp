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

    struct Pos{
        size_t x, y, z;

        ALPAKA_FN_HOST_ACC
        Pos() = default;

        ALPAKA_FN_HOST_ACC
        Pos(
            size_t const x_in,
            size_t const y_in,
            size_t const z_in
        ) :
            x( x_in ),
            y( y_in ),
            z( z_in )
        {}

        template<
          typename TDim,
          typename TSize,
          typename = typename std::enable_if<
              (TDim::value == 3u)
          >::type
        >
        ALPAKA_FN_HOST_ACC
        Pos(
            ::alpaka::Vec<
                TDim,
                TSize
            > const &vec
        )
        {
            for( uint32_t i(0); i < 3u; ++i ) {
                // alpaka vectors are z,y,x.
                ( &this->x )[ i ] = vec[ ( 3u - 1u ) - i ];
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
            > vec( x, y, z );
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
        cupla::Pos
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
        cupla::Pos
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
        cupla::Pos,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{

        ALPAKA_FN_HOST_ACC
        static auto
        getExtent( cupla::Pos const & extents )
        -> cupla::MemSizeType {
        return (&extents.x)[(3u - 1u) - T_Idx::value];
      }
    };

    //! extent set trait specialization
    template<
        typename T_Idx,
        typename T_Pos
    >
    struct SetExtent<
        T_Idx,
        cupla::Pos,
        T_Pos,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        setExtent(
            cupla::Pos &extents,
            T_Pos const &extent
        )
        -> void
        {
            (&extents.x)[(3u - 1u) - T_Idx::value] = extent;
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
        cupla::Pos,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        getOffset( cupla::Pos const & offsets )
        -> cupla::MemSizeType{
            return (&offsets.x)[(3u - 1u) - T_Idx::value];
        }
    };


    //! offset set trait specialization.
    template<
        typename T_Idx,
        typename T_Offset
    >
    struct SetOffset<
        T_Idx,
        cupla::Pos,
        T_Offset,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        setOffset(
            cupla::Pos &offsets,
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
        cupla::Pos
    >{
        using type = cupla::MemSizeType;
    };

} // namespace traits
} // namespace size
} // namespave alpaka
