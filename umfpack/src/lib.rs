#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![cfg_attr(feature = "strict", deny(warnings))]

extern crate rlst_suitesparse_src;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
