//! Demo the inverse of a matrix
use rlst::dense::linalg::elementary_matrix::ElementaryOperations;
pub use rlst::prelude::*;
use rlst::dense::linalg::elementary_matrix::{OpType, RowOpType};

pub fn main() {
    //Example 1: use elementary matrices to perform one step of an LU block decomposition
    //Here we use a 9x9 matrix and we partition it in 3x3 blocks
    let mut arr = rlst_dynamic_array2!(f64, [9, 9]);
    arr.fill_from_seed_equally_distributed(0);

    //We use the block 11 to eliminate block 21 using the LU with elementary matrices
    let mut block_11 = rlst_dynamic_array2!(f64, [3, 3]);
    let mut block_21 = rlst_dynamic_array2!(f64, [3, 3]);
    block_11.fill_from(arr.view().into_subview([0, 0], [3, 3]));
    block_21.fill_from(arr.view().into_subview([3, 0], [3, 3]));
    block_11.view_mut().into_inverse_alloc().unwrap();

    //We define the elementary matrix using the coordinates of the square blocks block_11 and block_21 and the dimension of the elementary matrix 
    //(in this case it's a 9x9 matrix and we assume that elementary matrices are squared)
    let mut el_mat_fact = empty_array().simple_mult_into_resize(block_21.view(), block_11.view());
    let row_indices: Vec<usize> = (3..6).collect();
    let col_indices: Vec<usize> = (0..3).collect();
    let el_mat = el_mat_fact.view_mut().into_el_mat_alloc(9, row_indices, col_indices, OpType::Row, false).unwrap();
    el_mat.el_mul(arr.view_mut(), Some(RowOpType::Sub), None).unwrap();
    
    //As a result we se that the block 21 has been eliminated
    let res = arr.view().into_subview([3, 0], [3, 3]);
    println!("L2 of block 21 after LU elimination through elementary matrices {}", res.view_flat().norm_2());


    //////////////////////////////////////////////////////////////////////////////////
    //Example 2: Use elementary matrices for row scaling 
    //Using the matrix of the previous example, we can re-scale the rows corresponding to blocks 11, 12, 13:
    let row_indices: Vec<usize> = (0..3).collect(); //The row and col indices must be the same, since the relevant dimension is the row_indices
    let col_indices: Vec<usize> = (0..3).collect();

    //We simply use a 1x1 matrix for this case, since the only relevant information is contained in the scaling factor
    let mut el_mat_fact = rlst_dynamic_array2!(f64, [1, 1]);
    let el_mat = el_mat_fact.view_mut().into_el_mat_alloc(9, row_indices, col_indices, OpType::Mul, false).unwrap();

    //We extract the elements of the matrix before scaling:
    let mut bef_scaling = rlst_dynamic_array2!(f64, [3, 9]);
    bef_scaling.fill_from(arr.view().into_subview([0, 0], [3, 9]));
    el_mat.el_mul(arr.view_mut(), None, Some(5.0)).unwrap();
    let aft_scaling = arr.view().into_subview([0, 0], [3, 9]);
    let norm_comp = aft_scaling.view_flat().norm_2()/bef_scaling.view_flat().norm_2();
    println!("The norm of the  rows corresponding to blocks 11, 12, 13 after scaling is {} times larger", norm_comp);

    
    //////////////////////////////////////////////////////////////////////////////////
    //Example 3: Use elementary matrices for row permutation
    // The permutation is as follows col_indices->row_indices. In this case:
    // row 0 moves into row 2, row 1 moves into row 0 and row 2 moves into row 1.
    let col_indices: Vec<usize> = [0, 1, 2].to_vec();
    let row_indices: Vec<usize> = [2, 0, 1].to_vec();
    
    let mut el_mat_fact = rlst_dynamic_array2!(f64, [1, 1]);
    let el_mat = el_mat_fact.view_mut().into_el_mat_alloc(9, row_indices, col_indices, OpType::Perm, false).unwrap();
    let mut bef_perm = rlst_dynamic_array2!(f64, [3, 9]);
    bef_perm.fill_from(arr.view().into_subview([0, 0], [3, 9]));
    el_mat.el_mul(arr.view_mut(), None, None).unwrap();
    let aft_perm = arr.view().into_subview([0, 0], [3, 9]);

    //Here we check if row 0 moved into row 2:
    let res = bef_perm.into_subview([0, 0], [1, 9]) - aft_perm.into_subview([2, 0], [1, 9]);

    println!("The difference between the row 0 before permutation and the row 2 after the permutation is {}",res.view_flat().norm_2());
}
