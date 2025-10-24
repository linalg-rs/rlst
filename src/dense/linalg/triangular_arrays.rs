//! Triangular matrix object.
use crate::dense::array::Array;
use crate::dense::traits::{
    RandomAccessByRef, RawAccessMut, Shape, Stride, UnsafeRandomAccessByValue,
};
use crate::dense::types::{c32, c64, RlstResult, RlstScalar};
use crate::{rlst_dynamic_array2, Side, TransMode, TriangularType};
use crate::{DynamicArray, RawAccess};
use blas::{ctrmm, ctrsm, dtrmm, dtrsm, strmm, strsm, ztrmm, ztrsm};

///Interface to obtain the upper-triangular part of the matrix
pub trait Triangular: RlstScalar {
    ///Compute the upper triangular
    fn into_triangular_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>
            + RandomAccessByRef<2, Item = Self>,
    >(
        arr: &Array<Self, ArrayImpl, 2>,
        triangular_type: TriangularType,
    ) -> RlstResult<TriangularMatrix<Self>>;
}

macro_rules! implement_into_triangular {
    ($scalar:ty) => {
        impl Triangular for $scalar {
            fn into_triangular_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>
                    + RandomAccessByRef<2, Item = Self>,
            >(
                arr: &Array<Self, ArrayImpl, 2>,
                triangular_type: TriangularType,
            ) -> RlstResult<TriangularMatrix<Self>> {
                TriangularMatrix::<$scalar>::new(arr, triangular_type)
            }
        }
    };
}

implement_into_triangular!(f32);
implement_into_triangular!(f64);
implement_into_triangular!(c32);
implement_into_triangular!(c64);

/// A struct representing an upper triangular matrix.
pub struct TriangularMatrix<Item: RlstScalar> {
    /// The upper triangular part of the matrix.
    pub tri: DynamicArray<Item, 2>,
    /// Defines if the matrix is lower or upper triangular
    pub triangular_type: TriangularType,
}

///Solver for upper-triangular matrix.
pub trait TriangularOperations: Sized {
    ///Defines an abstract type for the matrix to solve
    type Item: RlstScalar;
    ///Extract the upper triangular and save it
    fn new<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RandomAccessByRef<2, Item = Self::Item>,
    >(
        arr: &Array<Self::Item, ArrayImpl, 2>,
        triangular_type: TriangularType,
    ) -> RlstResult<Self>;
    ///Solves the upper-triangular system
    fn solve<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
    >(
        &self,
        arr_b: &mut Array<Self::Item, ArrayImpl, 2>,
        side: Side,
        trans: TransMode,
    );

    /// Multiplies the triangular matrix by a regular matrix
    fn mul<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
    >(
        &self,
        arr_b: &mut Array<Self::Item, ArrayImpl, 2>,
        side: Side,
        trans: TransMode,
    );
}

macro_rules! implement_solve_upper_triangular {
    ($scalar:ty, $trsm:expr, $trmm:expr) => {
        impl TriangularOperations for TriangularMatrix<$scalar> {
            type Item = $scalar;

            fn new<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
                    + Stride<2>
                    + Shape<2>
                    + RandomAccessByRef<2, Item = Self::Item>,
            >(
                arr: &Array<Self::Item, ArrayImpl, 2>,
                triangular_type: TriangularType,
            ) -> RlstResult<Self> {
                let shape = arr.shape();
                let mut tri = rlst_dynamic_array2!(Self::Item, shape);
                let mut view = tri.r_mut();

                match triangular_type {
                    TriangularType::Upper => {
                        for i in 0..shape[0] {
                            for j in i..shape[1] {
                                view[[i, j]] = arr[[i, j]];
                            }
                        }
                    }
                    TriangularType::Lower => {
                        for i in 0..shape[0] {
                            for j in 0..=i {
                                view[[i, j]] = arr[[i, j]];
                            }
                        }
                    }
                }
                Ok(Self {
                    tri,
                    triangular_type,
                })
            }

            fn solve<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self::Item>,
            >(
                &self,
                arr_b: &mut Array<$scalar, ArrayImpl, 2>,
                side: Side,
                trans: TransMode,
            ) {
                let lda = self.tri.stride()[1] as i32;
                let m = arr_b.shape()[0] as i32;
                let n = arr_b.shape()[1] as i32;
                let ldb = arr_b.stride()[1] as i32;
                let alpha = 1.0;

                let t_type_char = match self.triangular_type {
                    TriangularType::Upper => b'U', // A is on the left
                    TriangularType::Lower => b'L', // A is on the right
                };

                let side_char = match side {
                    Side::Left => b'L',  // A is on the left
                    Side::Right => b'R', // A is on the right
                };

                let trans_char = match trans {
                    TransMode::NoTrans => b'N',
                    TransMode::ConjNoTrans => {
                        panic!("TransMode::ConjNoTrans not supported for trsm implementation.")
                    }
                    TransMode::Trans => b'T',
                    TransMode::ConjTrans => b'C',
                };

                unsafe {
                    // dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
                    $trsm(
                        side_char,   // Side: A is on the left
                        t_type_char, // Uplo: A is upper triangular
                        trans_char,  // TransA: no transpose
                        b'N',        // Diag: A is not unit triangular
                        m,
                        n,
                        alpha.into(),
                        self.tri.data(),
                        lda,
                        arr_b.data_mut(),
                        ldb,
                    );
                }
            }

            fn mul<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self::Item>,
            >(
                &self,
                arr_b: &mut Array<$scalar, ArrayImpl, 2>,
                side: Side,
                trans: TransMode,
            ) {
                let lda = self.tri.stride()[1] as i32;
                let m = arr_b.shape()[0] as i32;
                let n = arr_b.shape()[1] as i32;
                let ldb = arr_b.stride()[1] as i32;
                let alpha = 1.0;

                let t_type_char = match self.triangular_type {
                    TriangularType::Upper => b'U', // A is on the left
                    TriangularType::Lower => b'L', // A is on the right
                };

                let side_char = match side {
                    Side::Left => b'L',  // A is on the left
                    Side::Right => b'R', // A is on the right
                };

                let trans_char = match trans {
                    TransMode::NoTrans => b'N',
                    TransMode::ConjNoTrans => {
                        panic!("TransMode::ConjNoTrans not supported for trsm implementation.")
                    }
                    TransMode::Trans => b'T',
                    TransMode::ConjTrans => b'C',
                };

                unsafe {
                    // dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
                    $trmm(
                        side_char,   // Side: A is on the left
                        t_type_char, // Uplo: A is upper triangular
                        trans_char,  // TransA: no transpose
                        b'N',        // Diag: A is not unit triangular
                        m,
                        n,
                        alpha.into(),
                        self.tri.data(),
                        lda,
                        arr_b.data_mut(),
                        ldb,
                    );
                }
            }
        }
    };
}

implement_solve_upper_triangular!(f32, strsm, strmm);
implement_solve_upper_triangular!(f64, dtrsm, dtrmm);
implement_solve_upper_triangular!(c32, ctrsm, ctrmm);
implement_solve_upper_triangular!(c64, ztrsm, ztrmm);
