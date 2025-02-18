# Rust Linear Solver Toolbox

The Rust Linear Solver toolbox is an in-development project for
dense and sparse linear algebra routines in Rust.

## Documentation
The latest documentation of the main branch of this repo is available at [linalg-rs.github.io/rlst](https://linalg-rs.github.io/rlst/).

## LICENSE

This work is dual-licensed under the Apache 2.0 and MIT license.
You can choose between one of them if you use this work.

`SPDX-License-Identifier: (Apache-2.0 OR MIT)`

Some optional dependencies of the library have different licenses that
may change the license of compiled library components. 

The [Suitesparse](https://people.engr.tamu.edu/davis/suitesparse.html) 
dependencies can be enabled with the `suitesparse`
feature flag. This enables AMD, CAMD, COLAMD, CCOLAMD, CHOLMOD, UMFPACK,
which are used to provide sparse direct solver
capabilities. UMFPACK is licensed under the GPL 2+ license, which affects any
code compiled against RLST with the `suitesparse` feature flag.

The [Sleef](https://sleef.org) dependency can be enabled with the `sleef` feature
flag. It is enabled by default and provides SIMD variants of certain mathematical functions.
Sleef is licensed under the Boost Software License Version 1.0.


## Notes

This library is the result of the merger of two experimental linear algebra projects

- Householder (github.com/UCL-ARC/householder)
- sandbox (github.com/linalg-rs/sandbox)

Both projects are MIT + Apache-2.0 dual licensed. The Rust Linear Solver
toolbox is the successor of both projects.
