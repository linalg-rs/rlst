//! Null space.
use crate::dense::array::Array;
use crate::dense::traits::{
    DefaultIterator, RawAccessMut, Shape, Stride, UnsafeRandomAccessByValue,
};
use crate::dense::types::{RlstResult, RlstScalar};
use crate::{
    empty_array, rlst_dynamic_array1, rlst_dynamic_array2, BaseArray, DynamicArray, MatrixQr,
    MatrixQrDecomposition, MatrixSvd, QrDecomposition, SvdMode, VectorContainer,
};
use itertools::min;
use num::Zero;

/// Compute the matrix nullspace.
///
/// The matrix nullspace is defined for a two dimensional array `arr` of
/// shape `[m, n]`.
///
/// # Example
///
/// The following command computes the nullspace of an array `a`.
/// The nullspace is found in
/// ```
/// # use rlst::rlst_dynamic_array2;
/// # use rlst::dense::linalg::null_space::{NullSpaceType, MatrixNull};
/// # let mut a = rlst_dynamic_array2!(f64, [3, 4]);
/// # a.fill_from_seed_equally_distributed(0);
/// # let null_res = a.r_mut().into_null_alloc(NullSpaceType::Row).unwrap();
/// ```
/// This method allocates memory for the nullspace computation.
pub trait MatrixNull: RlstScalar {
    /// Compute the matrix null space
    fn into_null_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        tol: <Self as RlstScalar>::Real,
        method: Method,
    ) -> RlstResult<NullSpace<Self>>
    where
        QrDecomposition<Self, ArrayImpl>: MatrixQrDecomposition<Item = Self>;
}

impl<T: RlstScalar + MatrixSvd + MatrixQr> MatrixNull for T {
    fn into_null_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        tol: <Self as RlstScalar>::Real,
        method: Method,
    ) -> RlstResult<NullSpace<Self>>
    where
        QrDecomposition<Self, ArrayImpl>: MatrixQrDecomposition<Item = Self>,
    {
        NullSpace::<Self>::new(arr, tol, method)
    }
}

impl<
        Item: RlstScalar + MatrixNull,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>,
    > Array<Item, ArrayImpl, 2>
{
    /// Compute the Column or Row nullspace of a given 2-dimensional array.
    pub fn into_null_alloc(
        self,
        tol: <Item as RlstScalar>::Real,
        method: Method,
    ) -> RlstResult<NullSpace<Item>>
    where
        QrDecomposition<Item, ArrayImpl>: MatrixQrDecomposition<Item = Item>,
    {
        <Item as MatrixNull>::into_null_alloc(self, tol, method)
    }
}

type RealScalar<T> = <T as RlstScalar>::Real;
/// Null Space
pub struct NullSpace<Item: RlstScalar> {
    ///Computed null space
    pub null_space_arr: Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>,
}

///Method defines the way to compute the null space
pub enum Method {
    ///SVD method
    Svd,
    ///QR method
    Qr,
}

///NullSpaceComputation creates the null space decomposition and saves it in the NullSpace Struct
pub trait NullSpaceComputation {
    ///This trait is implemented for RlstScalar (ie. f32, f64, c32, c64)
    type Item: RlstScalar;

    ///We create the null space decomposition
    fn new<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>,
    >(
        arr: Array<Self::Item, ArrayImpl, 2>,
        tol: RealScalar<Self::Item>,
        method: Method,
    ) -> RlstResult<Self>
    where
        Self: Sized,
        QrDecomposition<Self::Item, ArrayImpl>: MatrixQrDecomposition<Item = Self::Item>;
}

impl<T: RlstScalar + MatrixSvd + MatrixQr> NullSpaceComputation for NullSpace<T> {
    type Item = T;

    fn new<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>,
    >(
        arr: Array<Self::Item, ArrayImpl, 2>,
        tol: RealScalar<Self::Item>,
        method: Method,
    ) -> RlstResult<Self>
    where
        QrDecomposition<Self::Item, ArrayImpl>: MatrixQrDecomposition<Item = Self::Item>,
    {
        match method {
            Method::Svd => {
                let null_space_arr = svd_nullification(arr, tol);
                Ok(Self { null_space_arr })
            }
            Method::Qr => {
                let null_space_arr = qr_nullification(arr, tol);
                Ok(Self { null_space_arr })
            }
        }
    }
}

fn svd_nullification<
    Item: RlstScalar + MatrixSvd,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + RawAccessMut<Item = Item> + Shape<2>,
>(
    arr: Array<Item, ArrayImpl, 2>,
    tol: RealScalar<Item>,
) -> DynamicArray<Item, 2> {
    let shape: [usize; 2] = arr.shape();
    let dim: usize = min(shape).unwrap();
    let mut singular_values: DynamicArray<RealScalar<Item>, 1> =
        rlst_dynamic_array1!(RealScalar<Item>, [dim]);
    let mode: SvdMode = SvdMode::Full;
    let mut u: DynamicArray<Item, 2> = rlst_dynamic_array2!(Item, [shape[0], shape[0]]);
    let mut vt: DynamicArray<Item, 2> = rlst_dynamic_array2!(Item, [shape[1], shape[1]]);

    arr.into_svd_alloc(u.r_mut(), vt.r_mut(), singular_values.data_mut(), mode)
        .unwrap();

    //For a full rank rectangular matrix, then rank = dim.
    //find_matrix_rank checks if the matrix is full rank and recomputes the rank.
    let rank: usize = find_svd_rank::<Item>(&mut singular_values, dim, tol);

    //The null space is given by the last shape[1]-rank columns of V
    let mut null_space_arr: DynamicArray<Item, 2> = empty_array();
    null_space_arr.fill_from_resize(
        vt.conj()
            .transpose()
            .into_subview([0, rank], [shape[1], shape[1] - rank]),
    );

    null_space_arr
}

fn qr_nullification<
    Item: RlstScalar + MatrixQr,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + Shape<2> + RawAccessMut<Item = Item>,
>(
    arr: Array<Item, ArrayImpl, 2>,
    tol: RealScalar<Item>,
) -> DynamicArray<Item, 2>
where
    QrDecomposition<Item, ArrayImpl>: MatrixQrDecomposition<Item = Item>,
{
    let shape = arr.shape();
    let dim: usize = min(shape).unwrap();
    let qr = arr.into_qr_alloc().unwrap();
    //We compute the QR decomposition to find a linearly independent basis of the space.
    let mut q = rlst_dynamic_array2!(Item, [shape[0], shape[0]]);
    let _ = qr.get_q_alloc(q.r_mut());
    let mut r_mat: DynamicArray<Item, 2> = rlst_dynamic_array2!(Item, [dim, dim]);
    qr.get_r(r_mat.r_mut());
    //For a full rank rectangular matrix, then rank = dim.
    //find_matrix_rank checks if the matrix is full rank and recomputes the rank.
    let rank: usize = find_qr_rank::<Item>(&r_mat, dim, tol);
    let mut null_space_arr = empty_array();
    null_space_arr.fill_from_resize(q.into_subview([0, shape[1]], [shape[0], shape[0] - rank]));
    null_space_arr
}

fn find_svd_rank<Item: RlstScalar>(
    singular_values: &mut DynamicArray<RealScalar<Item>, 1>,
    dim: usize,
    tol: <Item as RlstScalar>::Real,
) -> usize {
    //We compute the rank of the matrix by expecting the values of the elements in the diagonal of R.
    let max: RealScalar<Item> = singular_values
        .r()
        .iter()
        .max_by(|a, b| (a.abs().partial_cmp(&b.abs())).unwrap())
        .unwrap()
        .abs();

    let rank: usize = if max.re() > <RealScalar<Item> as Zero>::zero() {
        let aux_vec: Vec<RealScalar<Item>> = singular_values
            .iter()
            .filter(|el| el.abs() > tol * max)
            .collect();
        aux_vec.len()
    } else {
        dim
    };

    rank
}

fn find_qr_rank<Item: RlstScalar>(
    r_mat: &DynamicArray<Item, 2>,
    dim: usize,
    tol: <Item as RlstScalar>::Real,
) -> usize {
    //We compute the rank of the matrix by expecting the values of the elements in the diagonal of R.
    let mut r_diag = rlst_dynamic_array1!(Item, [dim]);
    r_mat.get_diag(r_diag.r_mut());

    let max: RealScalar<Item> = r_diag
        .r()
        .iter()
        .max_by(|a, b| (a.abs().partial_cmp(&b.abs())).unwrap())
        .unwrap()
        .abs();

    let rank: usize = if max.re() > <RealScalar<Item> as Zero>::zero() {
        let aux_vec: Vec<Item> = r_diag.iter().filter(|el| el.abs() > tol * max).collect();
        aux_vec.len()
    } else {
        dim
    };

    rank
}
