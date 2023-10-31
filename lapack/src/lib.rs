//! Safe wrapper for Lapack

#![allow(clippy::too_many_arguments)]

use rlst_common::types::Scalar;

pub mod getrf;
pub mod getrs;

pub use getrf::Getrf;
pub use getrs::Getrs;

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum Trans {
    NoTranspose = b'N',
    Transpose = b'T',
    ConjugateTranspose = b'C',
}

// Collective Lapack wrapper trait
pub trait Lapack: Scalar + Getrf + Getrs {}

impl<T: Scalar + Getrf + Getrs> Lapack for T {}
