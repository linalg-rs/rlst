# Rust Linear Solver Toolbox

The Rust Linear Solver toolbox is an in-development project for
dense and sparse library routines in Rust.

# LICENSE

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


# Notes

This library is the result of the merger of two experimental linear algebra projects

- Householder (github.com/UCL-ARC/householder)
- sandbox (github.com/linalg-rs/sandbox)

Both projects are MIT + Apache-2.0 dual licensed. The Rust Linear Solver
toolbox is the successor of both projects.
