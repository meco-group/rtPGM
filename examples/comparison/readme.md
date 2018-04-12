# Comparison with state-of-the-art MPC solvers

This example compares the real-time PGM with 2 well-known solvers for embedded MPC: [qpOASES](https://projects.coin-or.org/qpOASES) and [HPMPC](https://github.com/giaf/hpmpc). In order to run this example, make sure you have a working install of [acados](https://github.com/acados/acados), which comes with the solvers. At the time of making the comparison, acados is in heavily development. The example was linked with commit [8625432](https://github.com/acados/acados/commit/8625432fee5ba00ce76d6b4823018f1ce22ddea4).

## Building the example
The example uses `cmake` to generate the build environment.

```
mkdir build
cd build
cmake ..
make
cd ..
```

## Running the example
```
./run_comparison.py
```
