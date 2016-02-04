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
#include <map>
#include <memory>

namespace cupla
{
namespace manager
{

    template<
        typename T_DeviceType
    >
    struct Device
    {
        using DeviceType = T_DeviceType;

        using DeviceMap = std::map<
            int,
            std::unique_ptr<
                DeviceType
            >
        >;

        DeviceMap m_map;
        int m_currentDevice;

        static Device &
        get()
        {
            static Device device;
            return device;
        }

        auto
        device(
            int idx = 0
        )
        -> DeviceType &
        {
            m_currentDevice = idx;
            auto iter = m_map.find( idx );
            if( iter != m_map.end() )
            {
                return *iter->second;
            }
            else
            {
                /* device id is not in the list
                 *
                 * select device with idx
                 *
                 * @todo check if device is available
                 */
                std::unique_ptr< DeviceType > dev(
                    new DeviceType(
                        ::alpaka::dev::DevMan<
                            DeviceType
                        >::getDevByIdx(
                            idx
                        )
                    )
                );
                m_map.insert(
                    std::make_pair( idx, std::move( dev ) )
                );
                return *m_map[ idx ];
            }
        }

        /**! reset the current device
         *
         * streams, memory and events on the current device must be
         * deleted at first by the user
         *
         * @return true in success case else false
         */
        bool reset()
        {
            ::alpaka::dev::reset( this->current( ) );
            auto iter = m_map.find( this->id( ) );

            if( iter == m_map.end() )
            {
                std::cerr << "device " << this->id( ) <<
                    " can not destroyed (was never created) " <<
                    std::endl;
                return false;
            }
            else
            {
                m_map.erase( iter );
                return true;
            }
        }

        auto
        id()
        -> int
        {
            return m_currentDevice;
        }

        auto
        current()
        -> DeviceType &
        {
            return this->device( this->id( ) );
        }

        auto
        count()
        -> int
        {
            return ::alpaka::dev::DevMan< DeviceType >::getDevCount( );
        }

    protected:
        Device() : m_currentDevice( 0 )
        {

        }

    };

} //namespace manager
} //namespace cupla
