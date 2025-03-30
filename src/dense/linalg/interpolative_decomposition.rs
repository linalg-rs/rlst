//! Demo the null space of a matrix.
use crate::dense::array::Array;
use crate::dense::traits::accessors::RandomAccessMut;
use crate::dense::traits::{
    DefaultIterator, MultIntoResize, RawAccessMut, Shape, Stride, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::dense::types::{c32, c64};
use crate::dense::types::{RlstResult, RlstScalar};
use crate::{empty_array, rlst_dynamic_array1, rlst_dynamic_array2, BaseArray, VectorContainer};
use num::One;

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
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        tol: <Self as RlstScalar>::Real,
        k: Option<usize>,
    ) -> RlstResult<IdDecomposition<Self, ArrayImpl>>;
}

macro_rules! implement_into_id {
    ($scalar:ty) => {
        impl MatrixId for $scalar {
            fn into_id_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>,
            >(
                arr: Array<Self, ArrayImpl, 2>,
                tol: <Self as RlstScalar>::Real,
                k: Option<usize>,
            ) -> RlstResult<IdDecomposition<Self, ArrayImpl>> {
                IdDecomposition::<$scalar, ArrayImpl>::new(arr, tol, k)
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
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>,
    > Array<Item, ArrayImplId, 2>
{
    /// Compute the interpolative decomposition of a given 2-dimensional array.
    pub fn into_id_alloc(
        self,
        tol: <Item as RlstScalar>::Real,
        k: Option<usize>,
    ) -> RlstResult<IdDecomposition<Item, ArrayImplId>> {
        <Item as MatrixId>::into_id_alloc(self, tol, k)
    }
}

/// Compute the matrix interpolative decomposition
pub trait MatrixIdDecomposition: Sized {
    /// Item type
    type Item: RlstScalar;
    /// Array implementaion
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;

    /// Create a new Interpolative Decomposition from a given array.
    fn new(
        arr: Array<Self::Item, Self::ArrayImpl, 2>,
        tol: <Self::Item as RlstScalar>::Real,
        k: Option<usize>,
    ) -> RlstResult<Self>;

    ///Compute the permutation matrix associated to the Interpolative Decomposition
    fn get_p<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        arr: Array<Self::Item, ArrayImplMut, 2>,
        perm: Vec<usize>,
    );
}

///Stores the relevant features regarding interpolative decomposition.
pub struct IdDecomposition<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + Shape<2> + RawAccessMut<Item = Item>,
> {
    /// arr: permuted array
    pub arr: Array<Item, ArrayImpl, 2>,
    /// perm_mat: permutation matrix associated to the interpolative decomposition
    pub perm_mat: Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>,
    /// rank: rank of the matrix associated to the interpolative decomposition for a given tolerance
    pub rank: usize,
    ///id_mat: interpolative matrix calculated for a given tolerance
    pub id_mat: Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>,
}

macro_rules! impl_id {
    ($scalar:ty) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > MatrixIdDecomposition for IdDecomposition<$scalar, ArrayImpl>
        {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl;

            fn new(
                mut arr: Array<$scalar, ArrayImpl, 2>,
                tol: <$scalar as RlstScalar>::Real,
                k: Option<usize>,
            ) -> RlstResult<Self> {
                //We assume that for a matrix of m rows and n columns, n>m, so we apply ID the transpose
                let mut arr_trans: Array<
                    $scalar,
                    BaseArray<$scalar, VectorContainer<$scalar>, 2>,
                    2,
                > = rlst_dynamic_array2!($scalar, [arr.shape()[1], arr.shape()[0]]);
                arr_trans.fill_from(arr.r().conj().transpose());

                //We compute the QR decomposition using rlst QR decomposition
                let mut arr_qr: Array<$scalar, BaseArray<$scalar, VectorContainer<$scalar>, 2>, 2> =
                    rlst_dynamic_array2!($scalar, [arr.shape()[1], arr.shape()[0]]);
                arr_qr.fill_from(arr.r().conj().transpose());
                let arr_qr_shape = arr_qr.shape();
                let qr = arr_qr.r_mut().into_qr_alloc().unwrap();

                //We obtain relevant parameters of the decomposition: the permutation induced by the pivoting and the R matrix
                let perm = qr.get_perm();
                let mut r_mat = rlst_dynamic_array2!($scalar, [arr_qr_shape[1], arr_qr_shape[1]]);
                qr.get_r(r_mat.r_mut());

                //The maximum rank is given by the number of columns of the transposed matrix
                let dim: usize = arr_qr_shape[1];
                let rank: usize;

                //The rank can be given a priori, in which case, we do not need to compute the rank using the tolerance parameter.
                match k {
                    Some(k) => rank = k,
                    None => {
                        //We extract the diagonal to calculate the rank of the matrix
                        let mut r_diag: Array<
                            $scalar,
                            BaseArray<$scalar, VectorContainer<$scalar>, 1>,
                            1,
                        > = rlst_dynamic_array1!($scalar, [dim]);
                        r_mat.get_diag(r_diag.r_mut());
                        let max: $scalar = r_diag
                            .iter()
                            .max_by(|a, b| a.abs().total_cmp(&b.abs()))
                            .unwrap()
                            .abs()
                            .into();

                        //We compute the rank of the matrix
                        if max.re() > 0.0 {
                            let alpha: $scalar = (1.0 / max) as $scalar;
                            r_diag.scale_inplace(alpha);
                            let aux_vec = r_diag
                                .iter()
                                .filter(|el| el.abs() > tol.into())
                                .collect::<Vec<_>>();
                            rank = aux_vec.len();
                        } else {
                            rank = dim;
                        }
                    }
                }

                let mut perm_mat = rlst_dynamic_array2!($scalar, [dim, dim]);
                Self::get_p(perm_mat.r_mut(), perm);

                let mut perm_arr =
                    empty_array::<$scalar, 2>().simple_mult_into_resize(perm_mat.r_mut(), arr.r());

                for col in 0..arr.shape()[1] {
                    for row in 0..arr.shape()[0] {
                        arr.data_mut()[col * arr_qr_shape[1] + row] =
                            *perm_arr.get_mut([row, col]).unwrap()
                    }
                }

                //In the case the matrix is full rank or we get a matrix of rank 0, then return the identity matrix.
                //If not, compute the Interpolative Decomposition matrix.
                if rank == 0 || rank >= dim {
                    let mut id_mat = rlst_dynamic_array2!($scalar, [dim, dim]);
                    id_mat.set_identity();
                    Ok(Self {
                        arr,
                        perm_mat,
                        rank,
                        id_mat,
                    })
                } else {
                    let shape: [usize; 2] = r_mat.shape();
                    let mut id_mat: Array<
                        $scalar,
                        BaseArray<$scalar, VectorContainer<$scalar>, 2>,
                        2,
                    > = rlst_dynamic_array2!($scalar, [dim - rank, rank]);

                    let mut k11: Array<
                        $scalar,
                        BaseArray<$scalar, VectorContainer<$scalar>, 2>,
                        2,
                    > = rlst_dynamic_array2!($scalar, [rank, rank]);
                    k11.fill_from(r_mat.r_mut().into_subview([0, 0], [rank, rank]));
                    k11.r_mut().into_inverse_alloc().unwrap();
                    let mut k12: Array<
                        $scalar,
                        BaseArray<$scalar, VectorContainer<$scalar>, 2>,
                        2,
                    > = rlst_dynamic_array2!($scalar, [rank, dim - rank]);
                    k12.fill_from(
                        r_mat
                            .r_mut()
                            .into_subview([0, rank], [rank, shape[1] - rank]),
                    );

                    let res = empty_array().simple_mult_into_resize(k11.r(), k12.r());
                    id_mat.fill_from(res.r().conj().transpose().r());
                    Ok(Self {
                        arr,
                        perm_mat,
                        rank,
                        id_mat,
                    })
                }
            }

            fn get_p<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>,
            >(
                mut arr: Array<$scalar, ArrayImplMut, 2>,
                perm: Vec<usize>,
            ) {
                let m = arr.shape()[0];

                arr.set_zero();
                for col in 0..m {
                    arr[[col, perm[col]]] = <$scalar as One>::one();
                }
            }
        }
    };
}

impl_id!(f64);
impl_id!(f32);
impl_id!(c32);
impl_id!(c64);
