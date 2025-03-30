//! Null space.
use crate::dense::array::Array;
use crate::dense::traits::{
    DefaultIterator, RawAccessMut, Shape, Stride, UnsafeRandomAccessByValue,
};
use crate::dense::types::{RlstResult, RlstScalar};
use crate::{
    empty_array, rlst_dynamic_array1, rlst_dynamic_array2, BaseArray, DynamicArray, MatrixSvd,
    SvdMode, VectorContainer,
};
use itertools::min;
use num::{One, Zero};

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
    ) -> RlstResult<NullSpace<Self>>;
}

impl<T: RlstScalar + MatrixSvd> MatrixNull for T {
    fn into_null_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        tol: <Self as RlstScalar>::Real,
    ) -> RlstResult<NullSpace<Self>> {
        NullSpace::<Self>::new(arr, tol)
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
    pub fn into_null_alloc(self, tol: <Item as RlstScalar>::Real) -> RlstResult<NullSpace<Item>> {
        <Item as MatrixNull>::into_null_alloc(self, tol)
    }
}

type RealScalar<T> = <T as RlstScalar>::Real;
/// Null Space
pub struct NullSpace<Item: RlstScalar> {
    ///Computed null space
    pub null_space_arr: Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>,
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
    ) -> RlstResult<Self>
    where
        Self: Sized;

    ///This function helps us to find the matrix rank
    fn find_matrix_rank(
        singular_values: &mut DynamicArray<RealScalar<Self::Item>, 1>,
        dim: usize,
        tol: RealScalar<Self::Item>,
    ) -> usize;
}

impl<T: RlstScalar + MatrixSvd> NullSpaceComputation for NullSpace<T> {
    type Item = T;

    fn new<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>,
    >(
        arr: Array<Self::Item, ArrayImpl, 2>,
        tol: RealScalar<Self::Item>,
    ) -> RlstResult<Self> {
        let shape: [usize; 2] = arr.shape();
        let dim: usize = min(shape).unwrap();
        let mut singular_values: DynamicArray<RealScalar<Self::Item>, 1> =
            rlst_dynamic_array1!(RealScalar<Self::Item>, [dim]);
        let mode: SvdMode = SvdMode::Full;
        let mut u: DynamicArray<Self::Item, 2> =
            rlst_dynamic_array2!(Self::Item, [shape[0], shape[0]]);
        let mut vt: DynamicArray<Self::Item, 2> =
            rlst_dynamic_array2!(Self::Item, [shape[1], shape[1]]);

        arr.into_svd_alloc(u.r_mut(), vt.r_mut(), singular_values.data_mut(), mode)
            .unwrap();

        //For a full rank rectangular matrix, then rank = dim.
        //find_matrix_rank checks if the matrix is full rank and recomputes the rank.
        let rank: usize = Self::find_matrix_rank(&mut singular_values, dim, tol);

        //The null space is given by the last shape[1]-rank columns of V
        let mut null_space_arr: DynamicArray<Self::Item, 2> = empty_array();
        null_space_arr.fill_from_resize(
            vt.conj()
                .transpose()
                .into_subview([0, rank], [shape[1], shape[1] - rank]),
        );

        Ok(Self { null_space_arr })
    }

    fn find_matrix_rank(
        singular_values: &mut DynamicArray<RealScalar<Self::Item>, 1>,
        dim: usize,
        tol: <Self::Item as RlstScalar>::Real,
    ) -> usize {
        //We compute the rank of the matrix by expecting the values of the elements in the diagonal of R.
        let max: RealScalar<Self::Item> = singular_values
            .r()
            .iter()
            .max_by(|a, b| (a.abs().partial_cmp(&b.abs())).unwrap())
            .unwrap()
            .abs();
        let mut rank: usize = dim;

        if max.re() > <RealScalar<Self::Item> as Zero>::zero() {
            let alpha: RealScalar<Self::Item> = <RealScalar<Self::Item> as One>::one() / max;
            singular_values.scale_inplace(alpha);
            let aux_vec: Vec<RealScalar<Self::Item>> = singular_values
                .iter()
                .filter(|el| el.abs() > tol)
                .collect::<Vec<RealScalar<Self::Item>>>();
            rank = aux_vec.len();
        }

        rank
    }
}
