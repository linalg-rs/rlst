use crate::dense::array::Array;
use crate::dense::traits::{RandomAccessMut, RawAccessMut, Shape, Stride, UnsafeRandomAccessByValue, UnsafeRandomAccessMut, UnsafeRandomAccessByRef, MultIntoResize, DefaultIterator};
use crate::dense::types::{c32, c64, RlstResult, RlstScalar};
use crate::{BaseArray, VectorContainer, rlst_dynamic_array1, empty_array, rlst_dynamic_array2};
use crate::DynamicArray;


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
/// # let res = a.view_mut().into_id_alloc(tol, None).unwrap();
/// ```

/// This method allocates memory for the interpolative decomposition.
pub trait MatrixId: RlstScalar {
    ///This method allocates space for ID
    fn into_id_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>
    >(
       arr: Array<Self, ArrayImpl, 2>, rank_param: Accuracy<<Self as RlstScalar>::Real>
    ) -> RlstResult<IdDecomposition<Self>>;
}

macro_rules! implement_into_id {
    ($scalar:ty) => {
        impl MatrixId for $scalar {
            fn into_id_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>
            >(
                arr: Array<Self, ArrayImpl, 2>, rank_param: Accuracy<<Self as RlstScalar>::Real>
            ) -> RlstResult<IdDecomposition<Self>> {
                IdDecomposition::<$scalar>::new(arr, rank_param)
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
    pub fn into_id_alloc(self, rank_param: Accuracy<<Item as RlstScalar>::Real>) -> RlstResult<IdDecomposition<Item>> {
        <Item as MatrixId>::into_id_alloc(self, rank_param)
    }
}

/// Compute the matrix interpolative decomposition
pub trait MatrixIdDecomposition: Sized {
    /// Item type
    type Item: RlstScalar;
    /// Create a new Interpolative Decomposition from a given array.
    fn new<
    ArrayImpl: UnsafeRandomAccessByValue<2, Item=Self::Item>
        + Stride<2>
        + Shape<2>
        + RawAccessMut<Item =Self::Item>>(arr: Array<Self::Item, ArrayImpl, 2>, rank_param: Accuracy<<Self::Item as RlstScalar>::Real>) -> RlstResult<Self>;

    /// Compute the rank of the decomposition from a given tolerance
    fn rank_from_tolerance<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(r_diag: Array<Self::Item, ArrayImplMut, 2>, tol: <Self::Item as RlstScalar>::Real)->usize;
    
    ///Compute the permutation matrix associated to the Interpolative Decomposition
    fn get_p<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self, arr:  Array<Self::Item, ArrayImplMut, 2>
    );



}

///Stores the relevant features regarding interpolative decomposition. 
pub struct IdDecomposition<
    Item: RlstScalar
> {
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
pub enum Accuracy<T>{
    /// Indicates that the rank of the decomposition will be computed from a given tolerance
    Tol(T),
    /// Indicates that the rank of the decomposition is given beforehand by the user
    FixedRank(usize),
    /// Computes the rank from the tolerance, and if this one is smaller than a user set range, then we stick to the user set range
    MaxRank(T, usize)
}


macro_rules! impl_id {
    ($scalar:ty) => {
        impl MatrixIdDecomposition for IdDecomposition<$scalar>
        {
            type Item = $scalar;

            fn new<
            ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                + Stride<2>
                + Shape<2>
                + RawAccessMut<Item = $scalar>,
                >(arr: Array<$scalar, ArrayImpl, 2>, rank_param: Accuracy<<$scalar as RlstScalar>::Real>) -> RlstResult<Self>{

                //We compute the QR decomposition using rlst QR decomposition
                let mut arr_qr: DynamicArray<$scalar, 2> = rlst_dynamic_array2!($scalar, [arr.shape()[1], arr.shape()[0]]);
                arr_qr.fill_from(arr.view().conj().transpose());
                let arr_qr_shape = arr_qr.shape();
                let qr = arr_qr.view_mut().into_qr_alloc().unwrap();
                
                //We obtain relevant parameters of the decomposition: the permutation induced by the pivoting and the R matrix
                let perm = qr.get_perm();
                let mut r_mat = rlst_dynamic_array2!($scalar, [arr_qr_shape[1], arr_qr_shape[1]]);
                qr.get_r(r_mat.view_mut());

                //The maximum rank is given by the number of columns of the transposed matrix
                let dim: usize = arr_qr_shape[1];
                let rank: usize;

                //The rank can be given a priori, in which case, we do not need to compute the rank using the tolerance parameter.

                match rank_param{
                    Accuracy::Tol(tol) =>{
                        rank = Self::rank_from_tolerance(r_mat.view_mut(), tol);
                    },
                    Accuracy::FixedRank(k) => rank = k,
                    Accuracy::MaxRank(tol, k) =>{
                        rank = std::cmp::max(k, Self::rank_from_tolerance(r_mat.view_mut(), tol));
                    },
                }

                //We permute arr to extract the columns belonging to the skeleton
                let mut permutation = rlst_dynamic_array2!($scalar, [arr_qr_shape[1], arr_qr_shape[1]]);
                permutation.set_zero();

                for (index, &elem) in perm.iter().enumerate() {
                    *permutation.get_mut([index, elem]).unwrap() = <$scalar as num::One>::one();
                }

                let perm_arr = empty_array::<$scalar, 2>()
                    .simple_mult_into_resize(permutation.view_mut(), arr.view());

                let mut skel = empty_array();
                skel.fill_from_resize(perm_arr.into_subview([0,0],[rank, arr_qr_shape[0]]));

                //In the case the matrix is full rank or we get a matrix of rank 0, then return the identity matrix.
                //If not, compute the Interpolative Decomposition matrix.
                if rank == 0 || rank >= dim{
                    let mut id_mat = rlst_dynamic_array2!($scalar, [dim, dim]);
                    id_mat.set_identity();
                    Ok(Self{skel, perm, rank, id_mat})
                }
                else{

                    let shape: [usize; 2] = [arr_qr_shape[1], arr_qr_shape[1]];
                    let mut id_mat: DynamicArray<$scalar, 2> = rlst_dynamic_array2!($scalar, [dim-rank, rank]);
                    
                    let mut k11: DynamicArray<$scalar, 2>  = rlst_dynamic_array2!($scalar, [rank, rank]);
                    k11.fill_from(r_mat.view_mut().into_subview([0, 0], [rank, rank]));
                    k11.view_mut().into_inverse_alloc().unwrap();
                    let mut k12: DynamicArray<$scalar, 2> = rlst_dynamic_array2!($scalar, [rank, dim-rank]);
                    k12.fill_from(r_mat.view_mut().into_subview([0, rank], [rank, shape[1]-rank]));

                    let res = empty_array().simple_mult_into_resize(k11.view(), k12.view());
                    id_mat.fill_from(res.view().conj().transpose().view());
                    Ok(Self{skel, perm, rank, id_mat})
                }
            }

            fn rank_from_tolerance<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>,
            >(r_mat: Array<$scalar, ArrayImplMut, 2>, tol: <$scalar as RlstScalar>::Real)->usize{
                let dim = r_mat.shape()[0];
                let mut r_diag:Array<$scalar, BaseArray<$scalar, VectorContainer<$scalar>, 1>, 1> = rlst_dynamic_array1!($scalar, [dim]);
                r_mat.get_diag(r_diag.view_mut());
                let max: $scalar = r_diag.iter().max_by(|a, b| a.abs().total_cmp(&b.abs())).unwrap().abs().into();

                //We compute the rank of the matrix
                if max.re() > 0.0{
                    let alpha: $scalar = (1.0/max) as $scalar;
                    r_diag.scale_inplace(alpha);
                    let aux_vec = r_diag.iter().filter(|el| el.abs() > tol.into() ).collect::<Vec<_>>();
                    aux_vec.len()
                }
                else{
                    dim
                }
            }

            fn get_p<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>,
            >(
                &self, mut arr: Array<$scalar, ArrayImplMut, 2>
            ) {
                arr.set_zero();
                for (index, &elem) in self.perm.iter().enumerate() {
                    *arr.get_mut([index, elem]).unwrap() = <$scalar as num::One>::one();
                }
            }
        }
    };
}

impl_id!(f64);
impl_id!(f32);
impl_id!(c32);
impl_id!(c64);
