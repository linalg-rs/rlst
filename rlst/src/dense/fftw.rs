//! Implementation of fft using fftw

pub mod c2c_inplace;

use std::ffi::c_void;

use crate::{
    Array, RawAccessMut, RlstScalar, Shape, Stride,
    dense::array::reference::{ArrayRef, ArrayRefMut},
};
use fftw_sys;

/// The direction of the FFT
#[derive(Copy, Clone, Debug)]
#[repr(i32)]
pub enum FftDirection {
    /// Forward FFT
    Forward = -1,
    /// Inverse FFT
    Backward = 1,
}

/// FFT Precision
#[derive(Copy, Clone, Debug)]
pub enum FftPrecision {
    /// Single Precision FFT
    Single,
    /// Double Precision FFT
    Double,
}

/// Flags for FFT Plans
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum FftPlanFlags {
    /// Use a simple heuristic to select the plan. Does not overwrite data.
    Estimate = fftw_sys::FFTW_ESTIMATE,
    /// Compute several FFTs and measure execution time. Overwrites data.
    Measure = fftw_sys::FFTW_MEASURE,
    /// Like [FftPlanFlags::Measure] but selects from a wider set of algorithms.
    Patient = fftw_sys::FFTW_PATIENT,
    /// Even wider search than [FftPlanFlags::Patient]
    Exhaustive = fftw_sys::FFTW_EXHAUSTIVE,
    /// Allow out of place transform to overwrite the input.
    DestroyInput = fftw_sys::FFTW_DESTROY_INPUT,
    /// Only use wisdowm to generate a plan. Do not return a plan if there is no wisdom.
    WisdomOnly = fftw_sys::FFTW_WISDOM_ONLY,
    /// Out-of-place transforms must not overwrite input (not available for all plan types).
    Unaligned = fftw_sys::FFTW_UNALIGNED,
}

/// Description of a plan with different input and output arrays.
pub trait FftPlan<const NDIM: usize> {
    /// The input type of the plan
    type ItemIn: RlstScalar;

    /// The output type of the plan
    type ItemOut: RlstScalar;

    /// Execute the plan for a forward FFT
    fn execute_forward(&mut self);

    /// Execute the plan for an inverse FFT
    fn execute_backwawrd(&mut self);
}

/// Storage for a pointer to an FFT Plan
struct FftPlanPtr {
    precision: FftPrecision,
    ptr: *mut c_void,
}

impl Drop for FftPlanPtr {
    fn drop(&mut self) {
        match self.precision {
            FftPrecision::Double => unsafe {
                fftw_sys::fftw_destroy_plan(self.ptr as fftw_sys::fftw_plan);
            },
            FftPrecision::Single => unsafe {
                fftw_sys::fftwf_destroy_plan(self.ptr as fftw_sys::fftwf_plan);
            },
        }
    }
}
