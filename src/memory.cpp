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
#include "cupla/api/memory.hpp"


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

cuplaError_t
cuplaMallocPitch(
    void ** devPtr,
    size_t * pitch,
    size_t const width,
    size_t const height
)
{
    const ::alpaka::Vec<
        cupla::AlpakaDim< 2u >,
        cupla::MemSizeType
    > extent( height, width );

    auto& buf = cupla::manager::Memory<
        cupla::AccDev,
        cupla::AlpakaDim< 2u >
    >::get().alloc( extent );

    // @toto catch errors
    *devPtr = ::alpaka::mem::view::getPtrNative(buf);
    *pitch = ::alpaka::mem::view::getPitchBytes< 1u >( buf );

    return cuplaSuccess;
};

cuplaError_t
cuplaMalloc3D(
    cupla::PitchedPtr * const pitchedDevPtr,
    cupla::Extent const extent
)
{

    auto& buf = cupla::manager::Memory<
        cupla::AccDev,
        cupla::AlpakaDim< 3u >
    >::get().alloc( extent );

    // @toto catch errors
    *pitchedDevPtr = make_cuplaPitchedPtr(
        ::alpaka::mem::view::getPtrNative(buf),
        ::alpaka::mem::view::getPitchBytes< 2u >( buf ),
        extent.width,
        extent.height
    );

    return cuplaSuccess;
}

cupla::Extent
make_cuplaExtent(
    size_t const w,
    size_t const h,
    size_t const d
)
{
    return cupla::Extent( w, h, d );
}

cupla::PitchedPtr
make_cuplaPitchedPtr(
    void * const d,
    size_t const p,
    size_t const xsz,
    size_t const ysz
)
{
    return cupla::PitchedPtr( d, p, xsz, ysz );
}

cuplaError_t
cuplaMallocHost(
    void **ptrptr,
    size_t size
)
{
    const ::alpaka::Vec<
        cupla::AlpakaDim<1u>,
        cupla::MemSizeType
    > extent( size );

    auto& buf = cupla::manager::Memory<
        cupla::AccHost,
        cupla::AlpakaDim<1u>
    >::get().alloc( extent );

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    // only implemented if nvcc is used
    ::alpaka::mem::buf::pin( buf );
#endif

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
    else if(
        cupla::manager::Memory<
            cupla::AccDev,
            cupla::AlpakaDim<2u>
        >::get().free( ptr )
    )
        return cuplaSuccess;
    else if(
        cupla::manager::Memory<
            cupla::AccDev,
            cupla::AlpakaDim<3u>
        >::get().free( ptr )
    )
        return cuplaSuccess;
    else
        return cuplaErrorMemoryAllocation;

}

cuplaError_t cuplaFreeHost(void *ptr)
{

    if(
        cupla::manager::Memory<
            cupla::AccHost,
            cupla::AlpakaDim<1u>
        >::get().free( ptr )
    )
        return cuplaSuccess;
    else
        return cuplaErrorMemoryAllocation;

}

cuplaError_t cuplaMemcpyAsync(
    void *dst,
    const void *src,
    size_t count,
    enum cuplaMemcpyKind kind,
    cuplaStream_t stream
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

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( stream )
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
                streamObject,
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
                streamObject,
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
                streamObject,
                dDestBuf,
                dSrcBuf,
                numBytes
            );

        }
        break;
    }
    return cuplaSuccess;
}

cuplaError_t
cuplaMemcpy(
    void *dst,
    const void *src,
    size_t count,
    enum cuplaMemcpyKind kind
)
{
    cuplaDeviceSynchronize();

    cuplaMemcpyAsync(
        dst,
        src,
        count,
        kind,
        0
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( 0 )
    );
    ::alpaka::wait::wait( streamObject );

    return cuplaSuccess;
}

cuplaError_t
cuplaMemsetAsync(
    void * devPtr,
    int value,
    size_t count,
    cuplaStream_t stream
)
{
    auto& device(
        cupla::manager::Device<
            cupla::AccDev
        >::get().current()
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( stream )
    );

    ::alpaka::Vec<
        cupla::AlpakaDim<1u>,
        cupla::MemSizeType
    > const
    numBytes(count);

    cupla::DeviceBufWrapper< 1u >
    dBuf(
        static_cast< uint8_t * >( devPtr ),
        device,
        numBytes
    );

    ::alpaka::mem::view::set(
        streamObject,
        dBuf,
        value,
        numBytes
    );

    return cuplaSuccess;
}

cuplaError_t
cuplaMemset(
    void * devPtr,
    int value,
    size_t count
)
{
    cuplaDeviceSynchronize();

    cuplaMemsetAsync(
        devPtr,
        value,
        count,
        0
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( 0 )
    );
    ::alpaka::wait::wait( streamObject );

    return cuplaSuccess;
}

cuplaError_t
cuplaMemcpy2DAsync(
    void * dst,
    size_t const dPitch,
    void const * const src,
    size_t const sPitch,
    size_t const width,
    size_t const height,
    enum cuplaMemcpyKind kind,
    cuplaStream_t const stream
)
{
    const ::alpaka::Vec<
        cupla::AlpakaDim<2u>,
        cupla::MemSizeType
    > numBytes( height, width );

    const ::alpaka::Vec<
        cupla::AlpakaDim<2u>,
        cupla::MemSizeType
    > dstPitch( dPitch * height , dPitch );

    const ::alpaka::Vec<
        cupla::AlpakaDim<2u>,
        cupla::MemSizeType
    > srcPitch( sPitch * height , sPitch );

    auto& device(
        cupla::manager::Device<
            cupla::AccDev
        >::get().current()
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( stream )
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

            const cupla::HostBufWrapper< 2u > hBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                host,
                numBytes,
                srcPitch
            );
            cupla::DeviceBufWrapper< 2u > dBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                device,
                numBytes,
                dstPitch
            );

            ::alpaka::mem::view::copy(
                streamObject,
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
            const cupla::DeviceBufWrapper< 2u > dBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                device,
                numBytes,
                srcPitch
            );
            cupla::HostBufWrapper< 2u > hBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                host,
                numBytes,
                dstPitch
            );

            ::alpaka::mem::view::copy(
                streamObject,
                hBuf,
                dBuf,
                numBytes
            );

        }
            break;
        case cuplaMemcpyDeviceToDevice:
        {
            const cupla::DeviceBufWrapper< 2u > dSrcBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                device,
                numBytes,
                srcPitch
            );
            cupla::DeviceBufWrapper< 2u > dDestBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                device,
                numBytes,
                dstPitch
            );

            ::alpaka::mem::view::copy(
                streamObject,
                dDestBuf,
                dSrcBuf,
                numBytes
            );

        }
        break;
    }
    return cuplaSuccess;
}

cuplaError_t
cuplaMemcpy2D(
    void * dst,
    size_t const dPitch,
    void const * const src,
    size_t const sPitch,
    size_t const width,
    size_t const height,
    enum cuplaMemcpyKind kind
)
{
    cuplaDeviceSynchronize();

    cuplaMemcpy2DAsync(
        dst,
        dPitch,
        src,
        sPitch,
        width,
        height,
        kind,
        0
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( 0 )
    );
    ::alpaka::wait::wait( streamObject );

    return cuplaSuccess;
}
