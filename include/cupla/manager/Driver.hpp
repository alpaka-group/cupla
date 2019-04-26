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

#include "cupla/namespace.hpp"
#include "cupla/types.hpp"
#include "cupla/manager/Device.hpp"
#include "cupla/manager/Event.hpp"
#include "cupla/manager/Memory.hpp"
#include "cupla/manager/Stream.hpp"

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
namespace manager
{

/** initialize the cupla environment
 *
 * avoid side effects with singletons in the user code
 */
class Driver
{

public:
    static Driver& get()
    {
        static Driver driver;
        return driver;
    }
private:

    Driver()
    {
        cupla::manager::Device< cupla::AccDev >::get( );

        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get();

        cupla::manager::Memory<
            cupla::AccDev,
            cupla::AlpakaDim<3u>
        >::get();

        cupla::manager::Memory<
            cupla::AccDev,
            cupla::AlpakaDim<2u>
        >::get();

        cupla::manager::Memory<
            cupla::AccDev,
            cupla::AlpakaDim<1u>
        >::get();

        cupla::manager::Event<
            cupla::AccDev,
            cupla::AccStream
        >::get();
    }
};

} //namespace manager
} //namespace CUPLA_ACCELERATOR_NAMESPACE
} //namespace cupla
