Requirement to port your project to *cupla*
===========================================

- The build system must be `CMake`
- The code must compile able with a C++ compiler but can be written in C style

Reserved variable names
=======================

- Some variable names are forbidden to use on the host side
- Only allowed in kernel:
  - `blockDim`
  - `gridDim`
  - `elemDim` number of elements per thread (is an 3 dimensional struct)
  - `blockIdx`
  - `threadIdx`


Restrictions
============

Events with timing informations synchronize the stream where they were recorded.
Disable the timing information of the event be set the flag `cudaEventDisableTiming`
or `cuplaEventDisableTiming` while the event creation.


Poring step by step
===================

- Remove cuda specific includes on top of your header and source files
- Add include `cuda_to_cupla.hpp`

CUDA code
```C++
#include <cuda_runtime.h>
```
cupla code
```C++
/* This must be the first include.
 * The reason for this is that cupla renames cuda host functions and device build in 
 * variables by using macros and macro function.
 */
#include <cuda_to_cupla.hpp>
```

- Transform the kernel (__global__ function) to a functor
- Add the function prefix `ALPAKA_FN_ACC` to the `operator() const`
- The `operator()` must be qualified as `const`
- Add as first kernel parameter the accelerator with the name `acc`
  It is important that the accelerator is named `acc` because all
  cupla to alpaka replacements used the variable `acc`
- If the kernel calls other function you must pass the accelerator `acc` 
  to each call.
- Add the qualifier const to each parameter which is not changed inside the kernel

CUDA kernel
```C++
template< int blockSize >
__global__ void fooKernel( int * ptr, float value )
{
    ...
}
```
cupla kernel
```C++
template< int blockSize >
struct fooKernel
{
    template< typename T_Acc >
    ALPAKA_FN_ACC
    void operator()( T_Acc const & acc, int * const ptr, float const value) const
    {
        ...
    }
};
```

- The host side kernel call must be changed
Cuda host side kernel call
```C++
...
dim3 gridSize(42,1,1);
dim3 blockSize(256,1,1);
// extern shared memory and stream is optional
fooKernel< 16 ><<< gridSize, blockSize, 0, 0 >>>( ptr, 23 );
```

cupla host side kernel call
```C++
...
dim3 gridSize(42,1,1);
dim3 blockSize(256,1,1);
// extern shared memory and stream is optional
CUPLA_KERNEL(fooKernel< 16 >)( gridSize, blockSize, 0, 0 )( ptr, 23 );
```

- Static shared memory definitions

Cuda shared memory (in kernel)
```C++
...
__shared__ int foo;
__shared__ int fooCArray[32];
__shared__ int fooCArray2D[4][32];

// extern shared memory (size were defined while the host side kernel call)
extern __shared__ float fooPtr[];

int bar = fooCArray2D[ threadIdx.x ][ 0 ];
...
```
cupla shared memory (in kernel)
```C++
...
sharedMem( foo, int );
/* It is not possible to use shared memory C arrays in cupla
 * `cupla::Array<Type,size>` is equal to `std::Array` but `std::Array` is not supported
 *  in all accelerators.
 */
sharedMem( fooCArray, cupla::Array< int, 32 > );
sharedMem( fooCArray, cupla::Array< cupla::Array< int, 4 >, 32 > );

// extern shared memory (size were defined while the host side kernel call)
sharedMemExtern( fooPtr, float * );

int bar = fooCArray2D[ threadIdx.x ][ 0 ];
...
```
