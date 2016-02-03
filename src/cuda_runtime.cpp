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


#include "cupla_runtime.hpp"
#include "cupla/manager/Memory.hpp"
#include "cupla/manager/Device.hpp"
#include "cupla/manager/Stream.hpp"
#include "cupla/manager/Event.hpp"


cuplaError_t
cuplaSetDevice( int idx)
{
    cupla::manager::Device< cupla::AccDev >::get().device( idx );
    return cuplaSuccess;
}

cuplaError_t
cuplaGetDevice( int * deviceId )
{
    *deviceId = cupla::manager::Device< cupla::AccDev >::get().id();
    return cuplaSuccess;
}

cuplaError_t
cuplaEventCreate(
    cuplaEvent_t * event,
    unsigned int flags
)
{
    *event = cupla::manager::Event< 
        cupla::AccDev, 
        cupla::AccStream 
    >::get().create( flags );
    
    return cuplaSuccess;
};

cuplaError_t
cuplaEventDestroy( cuplaEvent_t event )
{
    if( 
        cupla::manager::Event< 
            cupla::AccDev, 
            cupla::AccStream 
        >::get().destroy( event ) 
    )
        return cuplaSuccess;
    else
        return cuplaErrorInitializationError;
};

cuplaError_t
cuplaEventRecord(
    cuplaEvent_t event,
    cuplaStream_t stream
)
{
    auto& streamObject = cupla::manager::Stream< 
        cupla::AccDev, 
        cupla::AccStream 
    >::get().stream( stream );
    auto& eventObject = cupla::manager::Event< 
        cupla::AccDev, 
        cupla::AccStream 
    >::get().event( event );
    
    eventObject.record( streamObject );
    return cuplaSuccess;
}

cuplaError_t
cuplaEventElapsedTime(
    float * ms,
    cuplaEvent_t start,
    cuplaEvent_t end
)
{
    auto& eventStart = cupla::manager::Event< 
        cupla::AccDev, 
        cupla::AccStream 
    >::get().event( start );
    auto& eventEnd = cupla::manager::Event< 
        cupla::AccDev, 
        cupla::AccStream 
    >::get().event( end );
    *ms = eventEnd.elapsedSince(eventStart);
    return cuplaSuccess;
}

cuplaError_t
cuplaEventSynchronize(
    cuplaEvent_t event
)
{
    auto& eventObject = cupla::manager::Event< 
        cupla::AccDev, 
        cupla::AccStream 
    >::get().event( event );
    ::alpaka::wait::wait( *eventObject );
    return cuplaSuccess;
}

cuplaError_t
cuplaMalloc(
    void **ptrptr,
    size_t size
)
{

    const ::alpaka::Vec<
        cupla::AlpakaDim<1u>, 
        cupla::MemSizeType
    > extent( size );

    auto& buf = cupla::manager::Memory<
        cupla::AccDev,
        cupla::AlpakaDim<1u>
    >::get().alloc( extent );

    // @toto catch errors
    *ptrptr = ::alpaka::mem::view::getPtrNative(buf);
    return cuplaSuccess;
}

cuplaError_t cuplaFree(void *ptr)
{

    if(
        cupla::manager::Memory<
            cupla::AccDev,
            cupla::AlpakaDim<1u>
        >::get().free( ptr )
    )
        return cuplaSuccess;
    else
        return cuplaErrorMemoryAllocation;

}

const char *
cuplaGetErrorString(cuplaError_t)
{
    return "cuplaGetErrorString is currently not supported\n";
}

cuplaError_t cuplaMemcpy(
    void *dst,
    const void *src,
    size_t count,
    enum cuplaMemcpyKind kind
) 
{
    const ::alpaka::Vec<
        cupla::AlpakaDim<1u>,
        cupla::MemSizeType
    > numBytes(count);

    auto& device( 
        cupla::manager::Device< 
            cupla::AccDev 
        >::get().current() 
    );

    auto& stream( 
        cupla::manager::Stream< 
            cupla::AccDev, 
            cupla::AccStream 
        >::get().stream( 0 ) 
    );

    switch(kind)
    {
        case cuplaMemcpyHostToDevice:
        {
            auto& host( 
                cupla::manager::Device< 
                    cupla::AccHost 
                >::get().current() 
            );

            const cupla::HostBufWrapper< 1u > hBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                host,
                numBytes
            );
            cupla::DeviceBufWrapper< 1u > dBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                device,
                numBytes
            );

            ::alpaka::mem::view::copy(
                stream,
                dBuf,
                hBuf,
                numBytes
            );

        } 
            break;
        case cuplaMemcpyDeviceToHost:
        {
            auto& host( 
                cupla::manager::Device< 
                    cupla::AccHost 
                >::get().current() 
            );
            const cupla::DeviceBufWrapper< 1u > dBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                device,
                numBytes
            );
            cupla::HostBufWrapper< 1u > hBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                host,
                numBytes
            );

            ::alpaka::mem::view::copy(
                stream,
                hBuf,
                dBuf,
                numBytes
            );

        } 
            break;
        case cuplaMemcpyDeviceToDevice:
        {
            const cupla::DeviceBufWrapper< 1u > dSrcBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                device,
                numBytes
            );
            cupla::DeviceBufWrapper< 1u > dDestBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                device,
                numBytes
            );

            ::alpaka::mem::view::copy(
                stream,
                dDestBuf,
                dSrcBuf,
                numBytes
            );

        } 
        break;
    }
    ::alpaka::wait::wait(stream);
    return cuplaSuccess;
  }

