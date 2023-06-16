#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

extern crate rlst_blis_src;
extern crate rlst_netlib_lapack_src;
extern crate rlst_suitesparse_src;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
