//! Safe wrapper for Lapack

#![allow(clippy::too_many_arguments)]

pub mod geqp3;
pub mod getrf;
pub mod getrs;
pub mod ormqr;
pub mod unmqr;

pub use geqp3::Geqp3;
pub use getrf::Getrf;
pub use getrs::Getrs;
pub use ormqr::Ormqr;
pub use unmqr::Unmqr;

// // Collective Lapack wrapper trait
// pub trait Lapack: Scalar + Getrf + Getrs + Unmqr + Ormqr {}

// impl<T: Scalar + Getrf + Getrs + Unmqr + Ormqr> Lapack for T {}
