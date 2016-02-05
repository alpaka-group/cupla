cupla Install Guide
======================

Requirements
------------

- **alpaka** 
  - `git@github.com:ComputationalRadiationPhysics/alpaka.git`
  - `git checkout dev`
  - `export ALPAKA_ROOT=<alpaka_SRC_CODE_DIR>`
  - for more information please read [README.md](https://github.com/ComputationalRadiationPhysics/alpaka/blob/master/README.md)

- **cupla** 
  - `git@github.com:ComputationalRadiationPhysics/cupla.git`
  - `export CUPLA_ROOT=<cupla_SRC_CODE_DIR>`

- **cmake**  2.8.12 or higher
  - *Debian/Ubuntu:* `sudo apt-get install cmake file cmake-curses-gui`
  - *Arch Linux:* `sudo pacman --sync cmake`


compile a example
-----------------

- create build directory `mkdir -p buildCuplaExample`
- `cd buildCuplaExample`
- `cmake $CUPLA_ROOT/example/CUDASamples/matrixMul -D<ACC_TYPE>=ON`
    - list of supported ACC_TYPES
        - `ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE`
        - `ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE`
        - `ALPAKA_ACC_GPU_CUDA_ENABLE`
        - `ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE` (only allowed in combination with
          `CUPLA_KERNEL_OPTI` and `CUPLA_KERNEL_ELEM`, because the `blockSize` must be dim3(1,1,1))
          see [TuningGuide.md](doc/TuningGuide.md)
- `make -j`
- `./matMul -wA=320 -wB=300 -hA=320 -hB=320`
