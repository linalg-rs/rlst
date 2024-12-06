use crate::dense::array::Array;
use crate::dense::traits::{RawAccessMut, Shape, Stride, UnsafeRandomAccessByValue, DefaultIterator};
use crate::dense::types::{c32, c64, RlstResult, RlstScalar};
use crate::{BaseArray, VectorContainer, rlst_dynamic_array1, rlst_dynamic_array2, empty_array};
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
pub trait MatrixNull: RlstScalar {
    /// Compute the matrix null space
    fn into_null_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        tol: <Self as RlstScalar>::Real
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
                tol: <Self as RlstScalar>::Real
            ) -> RlstResult<NullSpace<Self>> {
                NullSpace::<$scalar>::new(arr, tol)
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
    pub fn into_null_alloc(self, tol: <Item as RlstScalar>::Real) -> RlstResult<NullSpace<Item>> {
        <Item as MatrixNull>::into_null_alloc(self, tol)
    }
}


type RealScalar<T> = <T as RlstScalar>::Real;
/// QR decomposition
pub struct NullSpace<
    Item: RlstScalar
> {
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
        >(arr: Array<$scalar, ArrayImpl, 2>, tol: RealScalar<$scalar>) -> RlstResult<Self> {

                let shape = arr.shape();
                let dim: usize = min(shape).unwrap();
                let mut singular_values = rlst_dynamic_array1!(RealScalar<$scalar>, [dim]);
                let mode = crate::dense::linalg::svd::SvdMode::Full;
                let mut u = rlst_dynamic_array2!($scalar, [shape[0], shape[0]]);
                let mut vt = rlst_dynamic_array2!($scalar, [shape[1], shape[1]]);

                arr.into_svd_alloc(u.view_mut(), vt.view_mut(), singular_values.data_mut(), mode).unwrap();

                //For a full rank rectangular matrix, then rank = dim. 
                //find_matrix_rank checks if the matrix is full rank and recomputes the rank.
                let rank: usize = Self::find_matrix_rank(singular_values, dim, tol);

                //The null space is given by the last shape[1]-rank columns of V
                let mut null_space_arr = empty_array();
                null_space_arr.fill_from_resize(vt.conj().transpose().into_subview([0, rank], [shape[1], shape[1]-rank]));

                Ok(Self {null_space_arr})
       
            }

            fn find_matrix_rank(singular_values: Array<RealScalar<$scalar>, BaseArray<RealScalar<$scalar>, VectorContainer<RealScalar<$scalar>>, 1>, 1>, dim: usize, tol:<$scalar as RlstScalar>::Real)->usize{
                //We compute the rank of the matrix by expecting the values of the elements in the diagonal of R.
                let mut singular_values_copy: Array<RealScalar<$scalar>, BaseArray<RealScalar<$scalar>, VectorContainer<RealScalar<$scalar>>, 1>, 1> = rlst_dynamic_array1!(RealScalar<$scalar>, [dim]);
                singular_values_copy.fill_from(singular_values.view());

                let max: RealScalar<$scalar> = singular_values.view().iter().max_by(|a, b| a.abs().total_cmp(&b.abs())).unwrap().abs().into();
                let rank: usize;

                if max.re() > 0.0{
                    let alpha: RealScalar<$scalar> = (1.0/max);
                    singular_values_copy.scale_inplace(alpha);
                    let aux_vec = singular_values_copy.iter().filter(|el| el.abs() > tol ).collect::<Vec<_>>();
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

