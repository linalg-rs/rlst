//! Elementary matrices (row swapping, row multiplication and row addition)
use crate::dense::array::Array;
use crate::dense::traits::{RawAccessMut, Shape, Stride, UnsafeRandomAccessByValue, MultIntoResize};
use crate::dense::types::{c32, c64, RlstResult, RlstScalar, RlstError};
use crate::{empty_array, rlst_dynamic_array2};
use crate::dense::traits::accessors::RandomAccessMut;

///Allocate space for the elementary matrix
pub trait ElementaryMatrixData: RlstScalar {
    ///This is the method that allocates space for the elements of the elementary matrix
    fn into_el_mat_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>, dim:usize, row_indices: Vec<usize>, col_indices: Vec<usize>, op_type: OpType, trans: bool
    ) -> RlstResult<ElementaryMatrix<Self, ArrayImpl>>;
}

macro_rules! implement_into_el_mat {
    ($scalar:ty) => {
        impl ElementaryMatrixData for $scalar {
            fn into_el_mat_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>,
            >(
                arr: Array<Self, ArrayImpl, 2>, dim:usize, row_indices: Vec<usize>, col_indices: Vec<usize>, op_type: OpType, trans: bool
            ) -> RlstResult<ElementaryMatrix<Self, ArrayImpl>> {
                ElementaryMatrix::<$scalar, ArrayImpl>::new(arr, dim, row_indices, col_indices, op_type, trans)
            }
        }
    };
}

implement_into_el_mat!(f32);
implement_into_el_mat!(f64);
implement_into_el_mat!(c32);
implement_into_el_mat!(c64);

impl<
        Item: RlstScalar + ElementaryMatrixData,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>,
    > Array<Item, ArrayImpl, 2>
{
    ///This computes an elementary (row addition/substraction, row scaling and row permutation) matrix given a series of parameters
    pub fn into_el_mat_alloc(self, dim:usize, row_indices: Vec<usize>, col_indices: Vec<usize>, op_type: OpType, trans: bool) -> RlstResult<ElementaryMatrix<Item, ArrayImpl>> {
        <Item as ElementaryMatrixData>::into_el_mat_alloc(self, dim, row_indices, col_indices, op_type, trans)
    }
}

///These are the methods associated to elementary matrices
pub trait ElementaryOperations: Sized {
    /// Item type
    type Item: RlstScalar;
    /// Array implementaion
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;

    /// We create the Elementary matrix (E), so it can be applied to a matrix A. In other words, we implement E(A).
    /// arr: is needed when row additions/substractions on A are performed.
    /// dim: is the dimension of the elementary matrix, given by the row dimension of A.
    /// row_indices and col_indices are respectively the domain and range of this application; i.e. we take the rows of A
    /// corresponding to col_indices of A and we place the result of this operation in row_indices of A.
    /// op_type: indicates if we are adding/substracting rows of A, scaling rows of A by a scalar or if we are permuting rows of A.
    /// trans: indicates if we are applying E^T
    fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, dim:usize, row_indices: Vec<usize>, col_indices: Vec<usize>, op_type: OpType, trans: bool) -> RlstResult<Self>;
    
    ///This method performs E(A). Here:
    /// right_arr: matrix A.
    /// row_op_type: indicates substraction or addition of rows
    /// alpha: is the scaling parameter of a scaling is applied
    fn el_mul(self, right_arr: Array<Self::Item, Self::ArrayImpl, 2>, row_op_type: Option<RowOpType>, alpha: Option<Self::Item>)-> RlstResult<()> ;

    ///This method implements the row addition/substraction
    fn row_ops(self, right_arr: Array<Self::Item, Self::ArrayImpl, 2>, op_type: RowOpType);
    
    ///This method implements the row scaling
    fn row_mul(self, right_arr: Array<Self::Item, Self::ArrayImpl, 2>, alpha: Self::Item);

    ///This method implements the row permutation
    fn row_perm(self, right_arr: Array<Self::Item, Self::ArrayImpl, 2>);

}

/// Container for the LU Decomposition of a matrix.
pub struct ElementaryMatrix<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + Shape<2> + RawAccessMut<Item = Item>,
> {
    dim: usize,
    arr: Array<Item, ArrayImpl, 2>,
    row_indices: Vec<usize>, 
    col_indices: Vec<usize>,
    op_type: OpType,

    ///Indicates if the operation is to be transposed or not
    pub trans: bool

}

/// Type of row operation
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum RowOpType {
    /// Row addition
    Add = b'A',
    /// Row substraction
    Sub = b'S',
}

/// Type of Elementary matrix
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum OpType {
    /// Row operation (addition or substraction)
    Row = b'R',
    /// Row scaling
    Mul = b'M',
    /// Row permutation
    Perm = b'P',
}


macro_rules! impl_el_ops {
    ($scalar:ty) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > ElementaryOperations for ElementaryMatrix<$scalar, ArrayImpl>
        {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl;

            fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, dim:usize, row_indices: Vec<usize>, col_indices: Vec<usize>, op_type: OpType, trans: bool) -> RlstResult<Self> {
                
                match op_type{
                    OpType::Row => {},
                    OpType::Mul => {assert_eq!(row_indices.len(), col_indices.len())},
                    OpType::Perm => {assert_eq!(row_indices.len(), col_indices.len())}
                }

                Ok(Self{arr, dim, row_indices, col_indices, op_type, trans})
            }

            fn el_mul(self, right_arr: Array<Self::Item, Self::ArrayImpl, 2>, row_op_type: Option<RowOpType>, alpha: Option<Self::Item>)-> RlstResult<()> 
            {
                match self.op_type{
                    OpType::Row => {
                        match(row_op_type){
                            Some(row_op_type)=> Ok(self.row_ops(right_arr, row_op_type)),
                            None =>Err(RlstError::IoError("Missing parameter".to_string()))
                        }},
                    OpType::Mul => {
                        match(alpha){
                            Some(alpha) => Ok(self.row_mul(right_arr, alpha)),
                            None =>Err(RlstError::IoError("Missing parameter".to_string()))
                        }},
                    OpType::Perm => {Ok(self.row_perm(right_arr))}
                }
            }

            fn row_ops(self, mut right_arr: Array<Self::Item, Self::ArrayImpl, 2>, row_op_type: RowOpType){
                let mut subarr_cols = rlst_dynamic_array2!($scalar, [self.col_indices.len(), self.dim]);
                let mut subarr_rows = rlst_dynamic_array2!($scalar, [self.row_indices.len(), self.dim]);
                let mut aux_arr = empty_array();
                let col_indices :Vec<usize>;
                let row_indices :Vec<usize>;
                
                if self.trans{
                    aux_arr.fill_from_resize(self.arr.view().conj().transpose());
                    col_indices = self.row_indices;
                    row_indices = self.col_indices;
                }
                else{
                    aux_arr.fill_from_resize(self.arr.view());
                    col_indices = self.col_indices;
                    row_indices = self.row_indices;
                }

                let right_arr_shape = right_arr.view().shape();
                let dim = self.dim;


                for col in 0..dim{
                    for row in 0..col_indices.len(){
                        *subarr_cols.get_mut([row, col]).unwrap() = right_arr.data_mut()[col*right_arr_shape[0] + col_indices[row]];
                    }
                    for row in 0..row_indices.len(){
                        *subarr_rows.get_mut([row, col]).unwrap() = right_arr.data_mut()[col*right_arr_shape[0] + row_indices[row]];
                    }
                }

                let res_mul = empty_array::<$scalar, 2>().simple_mult_into_resize(aux_arr, subarr_cols.view_mut());
                let mut add_res = rlst_dynamic_array2!($scalar, subarr_rows.shape());

                match row_op_type{
                    RowOpType::Add => {add_res.fill_from(subarr_rows.view() + res_mul.view());},
                    RowOpType::Sub => {add_res.fill_from(subarr_rows.view() - res_mul.view());}
                }

                for col in 0..dim{
                    for row in 0..row_indices.len(){
                        right_arr.data_mut()[col*right_arr_shape[0] + row_indices[row]] = *add_res.get_mut([row, col]).unwrap();
                    }
                }
            }

            fn row_mul(self, mut right_arr: Array<Self::Item, Self::ArrayImpl, 2>, alpha: Self::Item){
                let right_arr_shape = right_arr.view().shape();
                let dim = self.dim;
                let row_indices = self.row_indices;

                for col in 0..dim{
                    for row in 0..row_indices.len(){
                        right_arr.data_mut()[col*right_arr_shape[0] + row_indices[row]] = alpha*right_arr.data_mut()[col*right_arr_shape[0] + row_indices[row]]
                    }
                }

            }

            fn row_perm(self, mut right_arr: Array<Self::Item, Self::ArrayImpl, 2>){
                let dim = self.dim;
                let col_indices :Vec<usize>;
                let row_indices :Vec<usize>;
                
                if self.trans{
                    col_indices = self.row_indices;
                    row_indices = self.col_indices;
                }
                else{
                    col_indices = self.col_indices;
                    row_indices = self.row_indices;
                }

                let right_arr_shape = right_arr.view().shape();
                let mut subarr_cols = rlst_dynamic_array2!($scalar, [col_indices.len(), dim]);

                for col in 0..dim{
                    for row in 0..col_indices.len(){
                        *subarr_cols.get_mut([row, col]).unwrap() = right_arr.data_mut()[col*right_arr_shape[0] + col_indices[row]];
                    }
                }

                for col in 0..dim{
                    for row in 0..row_indices.len(){
                        right_arr.data_mut()[col*right_arr_shape[0] + row_indices[row]] = *subarr_cols.get_mut([row, col]).unwrap();
                    }
                }
            }
        }
    };
}

impl_el_ops!(f64);
impl_el_ops!(f32);
impl_el_ops!(c64);
impl_el_ops!(c32);
