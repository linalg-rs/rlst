//! Interpolative decomposition of a matrix.
use crate::dense::array::Array;
use crate::dense::linalg::qr::MatrixQrDecomposition;
use crate::dense::traits::ResizeInPlace;
use crate::dense::traits::{
    MultIntoResize, RandomAccessByRef, RawAccessMut, Shape, Stride, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::dense::types::TransMode; // Import TransMode from the appropriate module
use crate::dense::types::{c32, c64, RlstResult, RlstScalar};
use crate::{empty_array, rlst_dynamic_array2, BaseArray, VectorContainer};
use crate::{DynamicArray, RawAccess};
use blas::{ctrsm, dtrsm, strsm, ztrsm};
/// Compute the matrix interpolative decomposition, by providing a rank and an interpolation matrix.
///
/// The matrix interpolative decomposition is defined for a two dimensional 'long' array `arr` of
/// shape `[m, n]`, where `n>m`.
///
/// # Example
///
/// The following command computes the interpolative decomposition of an array `a` for a given tolerance, tol.
/// ```
/// # use rlst::rlst_dynamic_array2;
/// # let tol: f64 = 1e-5;
/// # let mut a = rlst_dynamic_array2!(f64, [50, 100]);
/// # a.fill_from_seed_equally_distributed(0);
/// # let res = a.r_mut().into_id_alloc(tol, None).unwrap();
/// ```
/// This method allocates memory for the interpolative decomposition.
pub trait MatrixId: RlstScalar {
    ///This method allocates space for ID
    fn into_id_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + UnsafeRandomAccessMut<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        rank_param: Accuracy<<Self as RlstScalar>::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<IdDecomposition<Self>>;
}

macro_rules! implement_into_id {
    ($scalar:ty) => {
        impl MatrixId for $scalar {
            fn into_id_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + UnsafeRandomAccessMut<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>,
            >(
                arr: Array<Self, ArrayImpl, 2>,
                rank_param: Accuracy<<Self as RlstScalar>::Real>,
                trans_mode: TransMode,
            ) -> RlstResult<IdDecomposition<Self>> {
                IdDecomposition::<$scalar>::new(arr, rank_param, trans_mode)
            }
        }
    };
}

implement_into_id!(f32);
implement_into_id!(f64);
implement_into_id!(c32);
implement_into_id!(c64);

impl<
        Item: RlstScalar + MatrixId,
        ArrayImplId: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>,
    > Array<Item, ArrayImplId, 2>
{
    /// Compute the interpolative decomposition of a given 2-dimensional array.
    pub fn into_id_alloc(
        self,
        rank_param: Accuracy<<Item as RlstScalar>::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<IdDecomposition<Item>> {
        <Item as MatrixId>::into_id_alloc(self, rank_param, trans_mode)
    }
}

/// Compute the matrix interpolative decomposition
pub trait MatrixIdDecomposition: Sized {
    /// Item type
    type Item: RlstScalar;
    /// Create a new Interpolative Decomposition from a given array.
    fn new<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
    >(
        arr: Array<Self::Item, ArrayImpl, 2>,
        rank_param: Accuracy<<Self::Item as RlstScalar>::Real>,
        trans_mode: TransMode,
    ) -> RlstResult<Self>;

    /// Compute the rank of the decomposition from a given tolerance
    fn rank_from_tolerance<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        ut_mat: Array<Self::Item, ArrayImplMut, 2>,
        tol: <Self::Item as RlstScalar>::Real,
    ) -> usize;

    ///Compute the permutation matrix associated to the Interpolative Decomposition
    fn get_p<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplMut, 2>,
    );
}

///Stores the relevant features regarding interpolative decomposition.
pub struct IdDecomposition<Item: RlstScalar> {
    /// skel: skeleton of the interpolative decomposition
    pub skel: DynamicArray<Item, 2>,
    /// perm: permutation associated to the pivoting indiced interpolative decomposition
    pub perm: Vec<usize>,
    /// rank: rank of the matrix associated to the interpolative decomposition for a given tolerance
    pub rank: usize,
    ///id_mat: interpolative matrix calculated for a given tolerance
    pub id_mat: Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>,
}

///Options to decide the matrix rank
pub enum Accuracy<T> {
    /// Indicates that the rank of the decomposition will be computed from a given tolerance
    Tol(T),
    /// Indicates that the rank of the decomposition is given beforehand by the user
    FixedRank(usize),
    /// Computes the rank from the tolerance, and if this one is smaller than a user set range, then we stick to the user set range
    MaxRank(T, usize),
}

///Interface to obtain the upper-triangular part of the matrix
pub trait UpperTriangular: RlstScalar {
    ///Compute the upper triangular
    fn into_upper_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>
            + RandomAccessByRef<2, Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
    ) -> RlstResult<UpperTriangularMatrix<Self>>;
}

macro_rules! implement_into_upper_triangular {
    ($scalar:ty) => {
        impl UpperTriangular for $scalar {
            fn into_upper_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>
                    + RandomAccessByRef<2, Item = Self>,
            >(
                arr: Array<Self, ArrayImpl, 2>,
            ) -> RlstResult<UpperTriangularMatrix<Self>> {
                UpperTriangularMatrix::<$scalar>::new(arr)
            }
        }
    };
}

implement_into_upper_triangular!(f32);
implement_into_upper_triangular!(f64);
implement_into_upper_triangular!(c32);
implement_into_upper_triangular!(c64);

/// A struct representing an upper triangular matrix.
pub struct UpperTriangularMatrix<Item: RlstScalar> {
    /// The upper triangular part of the matrix.
    pub upper: DynamicArray<Item, 2>,
}

///Solver for upper-triangular matrix.
pub trait UppperTriangularOperations: Sized {
    ///Defines an abstract type for the matrix to solve
    type Item: RlstScalar;
    ///Extract the upper triangular and save it
    fn new<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + RandomAccessByRef<2, Item = Self::Item>,
    >(
        arr: Array<Self::Item, ArrayImpl, 2>,
    ) -> RlstResult<Self>;
    ///Solves the upper-triangular system
    fn solve_upper_triangular<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
    >(
        &self,
        arr_b: &mut Array<Self::Item, ArrayImpl, 2>,
    );
}

macro_rules! implement_solve_upper_triangular {
    ($scalar:ty, $trsm:expr) => {
        impl UppperTriangularOperations for UpperTriangularMatrix<$scalar> {
            type Item = $scalar;

            fn new<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self::Item>
                    + RandomAccessByRef<2, Item = Self::Item>,
            >(
                arr: Array<Self::Item, ArrayImpl, 2>,
            ) -> RlstResult<Self> {
                let shape = arr.shape();
                let mut upper = rlst_dynamic_array2!(Self::Item, shape);
                let mut view = upper.r_mut();
                for i in 0..shape[0] {
                    for j in i..shape[1] {
                        view[[i, j]] = arr[[i, j]];
                    }
                }
                Ok(Self { upper })
            }

            fn solve_upper_triangular<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self::Item>,
            >(
                &self,
                arr_b: &mut Array<$scalar, ArrayImpl, 2>,
            ) {
                let lda = self.upper.stride()[1] as i32;
                let m = arr_b.shape()[0] as i32;
                let n = arr_b.shape()[1] as i32;
                let ldb = arr_b.stride()[1] as i32;
                let alpha = 1.0;

                unsafe {
                    // dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
                    $trsm(
                        b'L', // Side: A is on the left
                        b'U', // Uplo: A is upper triangular
                        b'N', // TransA: no transpose
                        b'N', // Diag: A is not unit triangular
                        m,
                        n,
                        alpha.into(),
                        self.upper.data(),
                        lda,
                        arr_b.data_mut(),
                        ldb,
                    );
                }
            }
        }
    };
}

implement_solve_upper_triangular!(f32, strsm);
implement_solve_upper_triangular!(f64, dtrsm);
implement_solve_upper_triangular!(c32, ctrsm);
implement_solve_upper_triangular!(c64, ztrsm);

macro_rules! impl_id {
    ($scalar:ty) => {
        impl MatrixIdDecomposition for IdDecomposition<$scalar> {
            type Item = $scalar;

            fn new<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = Self::Item>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                arr: Array<$scalar, ArrayImpl, 2>,
                rank_param: Accuracy<<$scalar as RlstScalar>::Real>,
                trans_mode: TransMode,
            ) -> RlstResult<Self> {
                //We compute the QR decomposition using rlst QR decomposition
                let mut arr_work = empty_array();
                let mut u_tri = empty_array();

                match trans_mode {
                    TransMode::Trans => arr_work.fill_from_resize(arr.r().conj()),
                    TransMode::NoTrans => arr_work.fill_from_resize(arr.r().conj().transpose()),
                    TransMode::ConjNoTrans => arr_work.fill_from_resize(arr.r()),
                    TransMode::ConjTrans => arr_work.fill_from_resize(arr.r().transpose()),
                };

                let shape = arr_work.shape();
                u_tri.resize_in_place([shape[1], shape[1]]);
                let dim = shape[1];

                let qr = arr_work.r_mut().into_qr_alloc().unwrap();
                //We obtain relevant parameters of the decomposition: the permutation induced by the pivoting and the R matrix
                let perm = qr.get_perm();
                qr.get_r(u_tri.r_mut());

                //The maximum rank is given by the number of columns of the transposed matrix
                let rank: usize;

                //The rank can be given a priori, in which case, we do not need to compute the rank using the tolerance parameter.
                match rank_param {
                    Accuracy::Tol(tol) => {
                        rank = Self::rank_from_tolerance(u_tri.r_mut(), tol);
                    }
                    Accuracy::FixedRank(k) => rank = k,
                    Accuracy::MaxRank(tol, k) => {
                        rank = std::cmp::max(k, Self::rank_from_tolerance(u_tri.r_mut(), tol));
                    }
                }

                let mut permutation = rlst_dynamic_array2!($scalar, [shape[1], shape[1]]);
                permutation.set_zero();

                let mut view = permutation.r_mut();
                for (index, &elem) in perm.iter().enumerate() {
                    view[[index, elem]] = <$scalar as num::One>::one();
                }

                let mut perm_arr = empty_array::<$scalar, 2>();
                perm_arr.r_mut().mult_into_resize(
                    TransMode::NoTrans,
                    trans_mode,
                    num::One::one(),
                    permutation.r_mut(),
                    arr.r(),
                    num::Zero::zero(),
                );

                //We permute arr to extract the columns belonging to the skeleton
                let mut skel = empty_array();
                skel.fill_from_resize(perm_arr.into_subview([0, 0], [rank, shape[0]]));
                //In the case the matrix is full rank or we get a matrix of rank 0, then return the identity matrix.
                //If not, compute the Interpolative Decomposition matrix.
                if rank == 0 || rank >= dim {
                    let mut id_mat = rlst_dynamic_array2!($scalar, [dim, dim]);
                    id_mat.set_identity();
                    Ok(Self {
                        skel,
                        perm,
                        rank,
                        id_mat,
                    })
                } else {
                    let shape: [usize; 2] = [shape[1], shape[1]];
                    let mut id_mat: DynamicArray<$scalar, 2> =
                        rlst_dynamic_array2!($scalar, [dim - rank, rank]);
                    let r11 = UpperTriangularMatrix::<$scalar>::new(
                        u_tri.r_mut().into_subview([0, 0], [rank, rank]),
                    )
                    .unwrap();

                    let mut r12 = u_tri
                        .r_mut()
                        .into_subview([0, rank], [rank, shape[1] - rank]);
                    r11.solve_upper_triangular(&mut r12);

                    id_mat.fill_from(r12.r().conj().transpose().r());
                    Ok(Self {
                        skel,
                        perm,
                        rank,
                        id_mat,
                    })
                }
            }

            fn rank_from_tolerance<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>,
            >(
                ut_mat: Array<$scalar, ArrayImplMut, 2>,
                tol: <$scalar as RlstScalar>::Real,
            ) -> usize {
                let dim = ut_mat.shape()[0];
                let max = ut_mat.get([0, 0]).unwrap().abs();

                //We compute the rank of the matrix
                if max.re() > 0.0 {
                    let mut rank = 0;
                    for i in 0..dim {
                        if ut_mat.get([i, i]).unwrap().abs() > tol * max {
                            rank += 1;
                        }
                    }
                    rank
                } else {
                    dim
                }
            }

            fn get_p<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplMut, 2>,
            ) {
                arr.set_zero();
                let mut view = arr.r_mut();

                for (index, &elem) in self.perm.iter().enumerate() {
                    view[[index, elem]] = <$scalar as num::One>::one();
                }
            }
        }
    };
}

impl_id!(f64);
impl_id!(f32);
impl_id!(c32);
impl_id!(c64);
