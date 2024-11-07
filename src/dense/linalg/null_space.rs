use crate::dense::array::Array;
use crate::dense::traits::{
    RawAccessMut, Shape, Stride, UnsafeRandomAccessByValue,
};
use crate::dense::types::{c32, c64, RlstResult, RlstScalar};
use crate::{BaseArray, VectorContainer, rlst_dynamic_array1, rlst_dynamic_array2};
use crate::dense::traits::DefaultIterator;
use crate::empty_array;
use itertools::min;

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
/// # let null_res = a.view_mut().into_null_alloc(NullSpaceType::Row).unwrap();
/// ```

/// This method allocates memory for the nullspace computation.

// pub trait MatrixNull {
//     /// Compute the matrix null space
//     fn into_null_alloc(arr, null_space_type) -> RlstResult<NullSpace<Self>>;
// }

pub trait MatrixNull: RlstScalar {
    /// Compute the matrix null space
    fn into_null_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        null_space_type: NullSpaceType
    ) -> RlstResult<NullSpace<Self>>;
}

macro_rules! implement_into_null {
    ($scalar:ty) => {
        impl MatrixNull for $scalar {
            fn into_null_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>,
            >(
                arr: Array<Self, ArrayImpl, 2>,
                null_space_type: NullSpaceType
            ) -> RlstResult<NullSpace<Self>> {
                NullSpace::<$scalar>::new(arr, null_space_type)
            }
        }
    };
}

implement_into_null!(f32);
implement_into_null!(f64);
implement_into_null!(c32);
implement_into_null!(c64);

impl<
        Item: RlstScalar + MatrixNull,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>,
    > Array<Item, ArrayImpl, 2>
{
    /// Compute the Column or Row nullspace of a given 2-dimensional array.
    pub fn into_null_alloc(self, null_space_type: NullSpaceType) -> RlstResult<NullSpace<Item>> {
        <Item as MatrixNull>::into_null_alloc(self, null_space_type)
    }
}

///Null space
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum NullSpaceType {
    /// Row Nullspace
    Row = b'R',
    /// Column Nullspace
    Column = b'C',
}

/// QR decomposition
pub struct NullSpace<
    Item: RlstScalar
> {
    ///Row or column nullspace
    pub null_space_type: NullSpaceType,
    ///Computed null space
    pub null_space_arr: Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>,
}

macro_rules! implement_null_space {
    ($scalar:ty) => {
        impl NullSpace<$scalar>
        {
            fn new<
            ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                + Stride<2>
                + Shape<2>
                + RawAccessMut<Item = $scalar>,
        >(arr: Array<$scalar, ArrayImpl, 2>, null_space_type: NullSpaceType) -> RlstResult<Self> {

                let shape = arr.shape();

                match null_space_type {
                    NullSpaceType::Row => {
                        let mut arr_qr: Array<$scalar, BaseArray<$scalar, VectorContainer<$scalar>, 2>, 2> = rlst_dynamic_array2!($scalar, [shape[1], shape[0]]);
                        arr_qr.fill_from(arr.view().conj().transpose());
                        let mut null_space_arr = empty_array();
                        Self::find_null_space(arr_qr, &mut null_space_arr);
                        Ok(Self {null_space_type, null_space_arr})
        
                    },
                    NullSpaceType::Column => {
                        let mut null_space_arr = empty_array();
                        Self::find_null_space(arr, &mut null_space_arr);
                        Ok(Self {null_space_type, null_space_arr})
                    },
                }
       
            }

            fn find_null_space<
            ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                + Stride<2>
                + Shape<2>
                + RawAccessMut<Item = $scalar>,
        >(arr: Array<$scalar, ArrayImpl, 2>, null_space_arr: &mut Array<$scalar, BaseArray<$scalar, VectorContainer<$scalar>, 2>, 2>)
            {
                let shape = arr.shape();
                let dim: usize = min(shape).unwrap();
                let qr = arr.into_qr_alloc().unwrap();

                //We compute the QR decomposition to find a linearly independent basis of the space.
                let mut q: Array<$scalar, BaseArray<$scalar, VectorContainer<$scalar>, 2>, 2> = rlst_dynamic_array2!($scalar, [shape[0], shape[0]]);
                let _ = qr.get_q_alloc(q.view_mut());

                let mut r_mat = rlst_dynamic_array2!($scalar, [dim, dim]);
                qr.get_r(r_mat.view_mut());

                //For a full rank rectangular matrix, then rank = dim. 
                //find_matrix_rank checks if the matrix is full rank and recomputes the rank.
                let rank: usize = Self::find_matrix_rank(r_mat, dim);

                null_space_arr.fill_from_resize(q.into_subview([0, shape[1]], [shape[0], shape[0]-rank]));
            }

            fn find_matrix_rank(r_mat: Array<$scalar, BaseArray<$scalar, VectorContainer<$scalar>, 2>, 2>, dim: usize)->usize{
                //We compute the rank of the matrix by expecting the values of the elements in the diagonal of R.
                let mut r_diag:Array<$scalar, BaseArray<$scalar, VectorContainer<$scalar>, 1>, 1> = rlst_dynamic_array1!($scalar, [dim]);
                r_mat.get_diag(r_diag.view_mut());

                let max: $scalar = r_diag.iter().max_by(|a, b| a.abs().total_cmp(&b.abs())).unwrap().abs().into();
                let rank: usize;

                if max.re() > 0.0{
                    let alpha: $scalar = (1.0/max) as $scalar;
                    r_diag.scale_inplace(alpha);
                    let aux_vec = r_diag.iter().filter(|el| el.abs() > 1e-15 ).collect::<Vec<_>>();
                    rank = aux_vec.len();
                }
                else{
                    rank = dim;
                }
                rank

            }
        }
    };
}

implement_null_space!(f64);
implement_null_space!(f32);
implement_null_space!(c64);
implement_null_space!(c32);

