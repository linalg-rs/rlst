//! Basic types.

mod number_traits_impl;
mod rlst_num_impl;

pub use num::complex::Complex32 as c32;
pub use num::complex::Complex64 as c64;

use thiserror::Error;

use crate::traits::number_relations::{IsGreaterByOne, IsGreaterZero, IsSmallerByOne};
use crate::ContainerType;
use crate::ContainerTypeSelector;

/// The Rlst error type.
#[derive(Error, Debug)]
pub enum RlstError {
    /// Not implemented
    #[error("Method {0} is not implemented.")]
    NotImplemented(String),
    /// Operation failed
    #[error("Operation {0} failed.")]
    OperationFailed(String),
    /// Matrix is empty
    #[error("Matrix has empty dimension {0:#?}.")]
    MatrixIsEmpty((usize, usize)),
    /// Dimension mismatch
    #[error("Dimension mismatch. Expected {expected:}. Actual {actual:}")]
    SingleDimensionError {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },
    /// Index layout error
    #[error("Index Layout error: {0}")]
    IndexLayoutError(String),
    /// MPI rank error
    #[error("MPI Rank does not exist. {0}")]
    MpiRankError(i32),
    /// Incompatible stride for Lapack
    #[error("Incompatible stride for Lapack.")]
    IncompatibleStride,
    /// Lapack error
    #[error("Lapack error: {0}")]
    LapackError(#[from] LapackError),
    /// General error
    #[error("{0}")]
    GeneralError(String),
    /// I/O error
    #[error("I/O Error: {0}")]
    IoError(String),
    /// Umfpack error
    #[error("Umfpack Error Code: {0}")]
    UmfpackError(i32),
    /// Matrix is not square
    #[error("Matrix is not square. Dimension: {0}x{1}")]
    MatrixNotSquare(usize, usize),
    /// Matrix is not Hermitian
    #[error("Matrix is not Hermitian (complex conjugate symmetric).")]
    MatrixNotHermitian,
}

/// Alias for an Rlst Result type.
pub type RlstResult<T> = std::result::Result<T, RlstError>;

/// Transposition Mode.
#[derive(Clone, Copy, PartialEq)]
pub enum TransMode {
    /// No modification of matrix.
    NoTrans,
    /// Complex conjugate of matrix.
    ConjNoTrans,
    /// Transposition of matrix.
    Trans,
    /// Conjugate transpose of matrix.
    ConjTrans,
}

// The following is a workaround to allow for implied trait bounds on associated types.
// A full descripton can be found at: https://docs.rs/imply-hack/latest/imply_hack/index.html

/// Trait to specify associated bounds as super trait bounds.
/// This allows the bounds to be implied and not explicitly stated in the type signature.
/// See the [imply-hack](https://docs.rs/imply-hack/latest/imply_hack/index.html) crate for more
/// details.
pub trait Imply<T>: sealed::ImplyInner<T, Is = T> {}

impl<T, U> Imply<T> for U {}

mod sealed {
    pub trait ImplyInner<T> {
        type Is;
    }

    impl<T, U> ImplyInner<T> for U {
        type Is = T;
    }
}

/// Memory layout of an object
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Column major
    ColumnMajor,
    /// Row major
    RowMajor,
    /// Unknown
    Unknown,
}

/// The Rlst error type.
#[derive(Error, Debug)]
pub enum LapackError {
    /// Info code from LAPACK
    #[error("LAPACK error code: {0}")]
    LapackInfoCode(i32),
}

/// Alias for a Lapack Result type.
pub type LapackResult<T> = std::result::Result<T, LapackError>;

/// Empty type for number relations
pub struct NumberType<const N: usize>;

impl IsGreaterByOne<0> for NumberType<1> {}
impl IsGreaterByOne<1> for NumberType<2> {}
impl IsGreaterByOne<2> for NumberType<3> {}
impl IsGreaterByOne<3> for NumberType<4> {}
impl IsGreaterByOne<4> for NumberType<5> {}
impl IsGreaterByOne<5> for NumberType<6> {}
impl IsGreaterByOne<6> for NumberType<7> {}
impl IsGreaterByOne<7> for NumberType<8> {}
impl IsGreaterByOne<8> for NumberType<9> {}
impl IsGreaterByOne<9> for NumberType<10> {}
impl IsGreaterByOne<10> for NumberType<11> {}
impl IsGreaterByOne<11> for NumberType<12> {}
impl IsGreaterByOne<12> for NumberType<13> {}
impl IsGreaterByOne<13> for NumberType<14> {}
impl IsGreaterByOne<14> for NumberType<15> {}
impl IsGreaterByOne<15> for NumberType<16> {}
impl IsGreaterByOne<16> for NumberType<17> {}
impl IsGreaterByOne<17> for NumberType<18> {}
impl IsGreaterByOne<18> for NumberType<19> {}
impl IsGreaterByOne<19> for NumberType<20> {}

impl IsGreaterZero for NumberType<1> {}
impl IsGreaterZero for NumberType<2> {}
impl IsGreaterZero for NumberType<3> {}
impl IsGreaterZero for NumberType<4> {}
impl IsGreaterZero for NumberType<5> {}
impl IsGreaterZero for NumberType<6> {}
impl IsGreaterZero for NumberType<7> {}
impl IsGreaterZero for NumberType<8> {}
impl IsGreaterZero for NumberType<9> {}
impl IsGreaterZero for NumberType<10> {}
impl IsGreaterZero for NumberType<11> {}
impl IsGreaterZero for NumberType<12> {}
impl IsGreaterZero for NumberType<13> {}
impl IsGreaterZero for NumberType<14> {}
impl IsGreaterZero for NumberType<15> {}
impl IsGreaterZero for NumberType<16> {}
impl IsGreaterZero for NumberType<17> {}
impl IsGreaterZero for NumberType<18> {}
impl IsGreaterZero for NumberType<19> {}
impl IsGreaterZero for NumberType<20> {}

impl IsSmallerByOne<1> for NumberType<0> {}
impl IsSmallerByOne<2> for NumberType<1> {}
impl IsSmallerByOne<3> for NumberType<2> {}
impl IsSmallerByOne<4> for NumberType<3> {}
impl IsSmallerByOne<5> for NumberType<4> {}
impl IsSmallerByOne<6> for NumberType<5> {}
impl IsSmallerByOne<7> for NumberType<6> {}
impl IsSmallerByOne<8> for NumberType<7> {}
impl IsSmallerByOne<9> for NumberType<8> {}
impl IsSmallerByOne<10> for NumberType<9> {}
impl IsSmallerByOne<11> for NumberType<10> {}
impl IsSmallerByOne<12> for NumberType<11> {}
impl IsSmallerByOne<13> for NumberType<12> {}
impl IsSmallerByOne<14> for NumberType<13> {}
impl IsSmallerByOne<15> for NumberType<14> {}
impl IsSmallerByOne<16> for NumberType<15> {}
impl IsSmallerByOne<17> for NumberType<16> {}
impl IsSmallerByOne<18> for NumberType<17> {}
impl IsSmallerByOne<19> for NumberType<18> {}
impl IsSmallerByOne<20> for NumberType<19> {}

/// Determine whether a matrix is upper or lower triangular.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpLo {
    /// Upper triangular matrix.
    Upper,
    /// Lower triangular matrix.
    Lower,
}

// Container types.

/// An unknown container type.
pub struct Heap;

/// A static container with fixed size N.
pub struct Stack<const N: usize>;

impl ContainerType for Heap {
    const STR: &str = "Heap";
}
impl<const N: usize> ContainerType for Stack<N> {
    const STR: &str = "Stack";
}

/// Trait to select a container type based on the input types.
pub struct SelectContainerType;

impl ContainerTypeSelector<Heap, Heap> for SelectContainerType {
    type Type = Heap;
}

impl<const N: usize> ContainerTypeSelector<Stack<N>, Heap> for SelectContainerType {
    type Type = Stack<N>;
}

impl<const N: usize> ContainerTypeSelector<Heap, Stack<N>> for SelectContainerType {
    type Type = Stack<N>;
}

impl<const N: usize> ContainerTypeSelector<Stack<N>, Stack<N>> for SelectContainerType {
    type Type = Stack<N>;
}
