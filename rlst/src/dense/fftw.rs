//! Implementation of fft using fftw
//!
//! FFTW uses plans to execute an FFT computation. Within FFTW there are two steps
//! to get the FFT of a given array.
//!
//! 1. Generate a plan for the desired FFT
//! 2. Execute the FFT
//!
//! FFTW supports FFT computations on one, two, and three dimensional input arrays.
//!
//! ## Complex-to-Complex transform
//!
//! The following gives an example of computing the Complex-to-Complex transform
//! of a given array `arr`.
//! ```
//! use rlst::{rlst_dynamic_array, dense::fftw::FftwPlanFlags,
//!     dense::fftw::C2CInplaceFft, EvaluateObject, c64,
//!     assert_array_relative_eq};
//! use num::FromPrimitive;
//!
//! let shape = [3, 5, 2];
//! let mut arr = rlst_dynamic_array!(c64, shape | 16);
//! arr.fill_from_seed_equally_distributed(0);
//!
//! // We use `eval` to copy the data into a new array before generating the plan
//! // so that `arr` is not overwritten.
//! let mut plan = arr.eval().into_c2c_fft(FftwPlanFlags::Estimate).expect("Count not create plan.");
//!
//! let forward = plan.r_mut().fft().eval();
//! let backward = plan.r_mut().ifft().eval();
//!
//! assert_array_relative_eq!(backward, c64::from_usize(30).unwrap() * arr.r(), 1E-10);
//!
//! ```
//! The method [into_c2c_fft](Array::into_c2c_fft) generates the c2c plan.
//! One can then execute a forward or backward fft as shown above.
//! By default the FFT overwrites the original array. With the `eval` function
//! the result is written into a new array. Note that the array in `backward`
//! is scaled compared to the original array. This is the default behaviour of FFTW.
//!
//! The FFTW interface in RLST always generates a forward and a backward FFT plan and
//! stores both plans.
//!
//! The `plan` struct satisfies all the usual traits for mutable array access. Hence,
//! it can be used itself like any other array. The underlying data is that of the original array
//! for who a plan was computed.
//!
//! ## Thread Safety
//!
//! Plan generation is not thread-safe in FFTW. This is solved in `rlst` by hiding the plan
//! generation inside a mutex. However, if other libraries access FFTW at the same time as `rlst`
//! a race condition may occur. The execution of a plan to compute the forward or backward FFT
//! is thread-safe.

pub mod c2c_inplace;

use std::{
    ffi::c_void,
    marker::PhantomData,
    sync::{LazyLock, Mutex},
};

use crate::{Array, RlstScalar};
use fftw_sys;
use num::Complex;

pub use c2c_inplace::C2CInplaceFft;

static FFTW_PLAN_INTERFACE: LazyLock<Mutex<FftwPlanInterface>> =
    LazyLock::new(|| Mutex::new(FftwPlanInterface::default()));

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
pub enum FftwPlanFlags {
    /// Use a simple heuristic to select the plan. Does not overwrite data.
    Estimate = fftw_sys::FFTW_ESTIMATE,
    /// Compute several FFTs and measure execution time. Overwrites data.
    Measure = fftw_sys::FFTW_MEASURE,
    /// Like [FftwPlanFlags::Measure] but selects from a wider set of algorithms.
    Patient = fftw_sys::FFTW_PATIENT,
    /// Even wider search than [FftwPlanFlags::Patient]
    Exhaustive = fftw_sys::FFTW_EXHAUSTIVE,
    /// Allow out of place transform to overwrite the input.
    DestroyInput = fftw_sys::FFTW_DESTROY_INPUT,
    /// Only use wisdowm to generate a plan. Do not return a plan if there is no wisdom.
    WisdomOnly = fftw_sys::FFTW_WISDOM_ONLY,
    /// Out-of-place transforms must not overwrite input (not available for all plan types).
    Unaligned = fftw_sys::FFTW_UNALIGNED,
}

/// Description of a plan with different input and output arrays.
pub trait FftPlanInplace<const NDIM: usize> {
    /// The input type of the plan
    type Item: RlstScalar;

    /// The input array implementation
    type ArrayImpl;

    /// Execute the plan for a forward FFT
    fn execute_forward(&mut self);

    /// Execute the plan for an inverse FFT
    fn execute_backward(&mut self);
}

struct FftwPlanPtrType<T> {
    _marker: PhantomData<T>,
}

trait FftwPlanPtrTypeTrait {
    type PtrType;

    fn get_fftw_ptr(ptr: *mut c_void) -> Self::PtrType;
}

impl FftwPlanPtrTypeTrait for FftwPlanPtrType<f64> {
    type PtrType = fftw_sys::fftw_plan;

    fn get_fftw_ptr(ptr: *mut c_void) -> Self::PtrType {
        ptr as fftw_sys::fftw_plan
    }
}

impl FftwPlanPtrTypeTrait for FftwPlanPtrType<f32> {
    type PtrType = fftw_sys::fftwf_plan;

    fn get_fftw_ptr(ptr: *mut c_void) -> Self::PtrType {
        ptr as fftw_sys::fftwf_plan
    }
}

/// Storage for a pointer to an FFT Plan
struct FftwPlanPtr<T>
where
    T: 'static,
    FftwPlanPtrType<T>: FftwPlanPtrTypeTrait,
{
    ptr: *mut c_void,
    _marker: std::marker::PhantomData<T>,
}

trait PlanInterfaceTrait {
    type PlanPtr;

    type RealType;

    /// Destroy a given plan
    unsafe fn destroy_plan(&self, plan_ptr: Self::PlanPtr);

    /// Create a c2c plan
    unsafe fn create_c2c(
        &self,
        rank: i32,
        shape: *const i32,
        arr_in: *mut Complex<Self::RealType>,
        arr_out: *mut Complex<Self::RealType>,
        sign: i32,
        flags: u32,
    ) -> Self::PlanPtr;
}

impl<T> Drop for FftwPlanPtr<T>
where
    T: 'static,
    FftwPlanPtrType<T>: FftwPlanPtrTypeTrait,
{
    fn drop(&mut self) {
        if coe::is_same::<T, f64>() {
            let ptr = FftwPlanPtrType::<f64>::get_fftw_ptr(self.ptr);
            unsafe {
                FFTW_PLAN_INTERFACE
                    .lock()
                    .unwrap()
                    .fftw_double
                    .destroy_plan(ptr);
            };
        } else if coe::is_same::<T, f32>() {
            let ptr = FftwPlanPtrType::<f32>::get_fftw_ptr(self.ptr);
            unsafe {
                FFTW_PLAN_INTERFACE
                    .lock()
                    .unwrap()
                    .fftw_single
                    .destroy_plan(ptr)
            };
        }
    }
}

/// An interface to plan generation routines that will be hidden behind a mutex
struct ConcretePlanInterface<T> {
    _marker: PhantomData<T>,
}

impl<T> Default for ConcretePlanInterface<T> {
    fn default() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

impl PlanInterfaceTrait for ConcretePlanInterface<f64> {
    type PlanPtr = fftw_sys::fftw_plan;

    type RealType = f64;

    unsafe fn destroy_plan(&self, plan_ptr: Self::PlanPtr) {
        unsafe {
            fftw_sys::fftw_destroy_plan(plan_ptr);
        }
    }

    unsafe fn create_c2c(
        &self,
        rank: i32,
        shape: *const i32,
        arr_in: *mut Complex<Self::RealType>,
        arr_out: *mut Complex<Self::RealType>,
        sign: i32,
        flags: u32,
    ) -> Self::PlanPtr {
        unsafe { fftw_sys::fftw_plan_dft(rank, shape, arr_in, arr_out, sign, flags) }
    }
}

impl PlanInterfaceTrait for ConcretePlanInterface<f32> {
    type PlanPtr = fftw_sys::fftwf_plan;

    type RealType = f32;

    unsafe fn destroy_plan(&self, plan_ptr: Self::PlanPtr) {
        unsafe {
            fftw_sys::fftwf_destroy_plan(plan_ptr);
        }
    }

    unsafe fn create_c2c(
        &self,
        rank: i32,
        shape: *const i32,
        arr_in: *mut Complex<Self::RealType>,
        arr_out: *mut Complex<Self::RealType>,
        sign: i32,
        flags: u32,
    ) -> Self::PlanPtr {
        unsafe { fftw_sys::fftwf_plan_dft(rank, shape, arr_in, arr_out, sign, flags) }
    }
}

/// A thread-safe interface to work with FFTW Plans
#[derive(Default)]
struct FftwPlanInterface {
    pub fftw_single: ConcretePlanInterface<f32>,
    pub fftw_double: ConcretePlanInterface<f64>,
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: FftPlanInplace<NDIM>,
{
    /// Compute the forward fft
    pub fn fft(mut self) -> Self {
        self.imp_mut().execute_forward();
        self
    }

    /// Compute the inverse fft
    pub fn ifft(mut self) -> Self {
        self.imp_mut().execute_backward();
        self
    }
}
