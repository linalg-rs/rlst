//! Interface to batched gemm operations

use crate::dense::array::DynamicArray;
use crate::dense::base_array::BaseArray;
use crate::dense::traits::{RawAccess, RawAccessMut, Shape, Stride};
use crate::dense::types::RlstScalar;
use crate::dense::types::TransMode;
use crate::{rlst_dynamic_array2, MultInto, RlstResult, SliceArray, SliceArrayMut};

/// Batched matrix-matrix products.
///
/// Implementations of this trait allow batched matrix-matrix products.
pub trait BatchedGemm {
    /// The scalar type.
    type Item: RlstScalar;

    /// Access the left matrix with given index.
    fn left_matrix(&self, index: usize) -> Option<SliceArray<'_, Self::Item, 2>>;

    /// Mutably access the left matrix with given index/
    fn left_matrix_mut(&mut self, index: usize) -> Option<SliceArrayMut<'_, Self::Item, 2>>;

    /// Access the right matrix with given index.
    fn right_matrix(&self, index: usize) -> Option<SliceArray<'_, Self::Item, 2>>;

    /// Mutably access the right matrix with given index.
    fn right_matrix_mut(&mut self, index: usize) -> Option<SliceArrayMut<'_, Self::Item, 2>>;

    /// Access the result matrix with given index.
    fn result_matrix(&self, index: usize) -> Option<SliceArray<'_, Self::Item, 2>>;

    /// Mutably access the result matrix with given index.
    fn result_matrix_mut(&mut self, index: usize) -> Option<SliceArrayMut<'_, Self::Item, 2>>;

    /// Evaluate the batched matrix product.
    fn evaluate(&mut self) -> RlstResult<()>;
}

/// Batched matrix multiplication on CPU.
///
/// This implementation simply uses the available BLAS to perform
/// each matrix matrix multiplication.
pub struct DefaultCpuBatchedGemm<Item: RlstScalar> {
    left_matrices: Vec<DynamicArray<Item, 2>>,
    right_matrices: Vec<DynamicArray<Item, 2>>,
    result_matrices: Vec<DynamicArray<Item, 2>>,
    number_of_matrices: usize,
    alpha: Item,
    beta: Item,
}

impl<Item: RlstScalar> DefaultCpuBatchedGemm<Item> {
    /// Initialize a new COPU batched matrix multiplication.
    pub fn new(
        left_dim: (usize, usize),
        right_dim: (usize, usize),
        number_of_matrices: usize,
        alpha: Item,
        beta: Item,
    ) -> Self {
        assert_eq!(left_dim.1, right_dim.0);

        let mut left_matrices = Vec::<DynamicArray<Item, 2>>::with_capacity(number_of_matrices);
        let mut right_matrices = Vec::<DynamicArray<Item, 2>>::with_capacity(number_of_matrices);
        let mut result_matrices = Vec::<DynamicArray<Item, 2>>::with_capacity(number_of_matrices);

        for _ in 0..number_of_matrices {
            left_matrices.push(rlst_dynamic_array2!(Item, [left_dim.0, left_dim.1]));
            right_matrices.push(rlst_dynamic_array2!(Item, [right_dim.0, right_dim.1]));
            result_matrices.push(rlst_dynamic_array2!(Item, [left_dim.0, right_dim.1]));
        }

        Self {
            left_matrices,
            right_matrices,
            result_matrices,
            number_of_matrices,
            alpha,
            beta,
        }
    }
}

impl<Item: RlstScalar> BatchedGemm for DefaultCpuBatchedGemm<Item> {
    type Item = Item;

    fn left_matrix(&self, index: usize) -> Option<SliceArray<'_, Self::Item, 2>> {
        self.left_matrices.get(index).map(|mat| {
            let slice_container = crate::SliceContainer::new(mat.data());
            SliceArray::new(BaseArray::new_with_stride(
                slice_container,
                mat.shape(),
                mat.stride(),
            ))
        })
    }

    fn left_matrix_mut(&mut self, index: usize) -> Option<SliceArrayMut<'_, Self::Item, 2>> {
        self.left_matrices.get_mut(index).map(|mat| {
            let shape = mat.shape();
            let stride = mat.stride();
            let slice_container = crate::SliceContainerMut::new(mat.data_mut());
            SliceArrayMut::new(BaseArray::new_with_stride(slice_container, shape, stride))
        })
    }

    fn right_matrix(&self, index: usize) -> Option<SliceArray<'_, Self::Item, 2>> {
        self.right_matrices.get(index).map(|mat| {
            let slice_container = crate::SliceContainer::new(mat.data());
            SliceArray::new(BaseArray::new_with_stride(
                slice_container,
                mat.shape(),
                mat.stride(),
            ))
        })
    }

    fn right_matrix_mut(&mut self, index: usize) -> Option<SliceArrayMut<'_, Self::Item, 2>> {
        self.right_matrices.get_mut(index).map(|mat| {
            let shape = mat.shape();
            let stride = mat.stride();
            let slice_container = crate::SliceContainerMut::new(mat.data_mut());
            SliceArrayMut::new(BaseArray::new_with_stride(slice_container, shape, stride))
        })
    }

    fn result_matrix(&self, index: usize) -> Option<SliceArray<'_, Self::Item, 2>> {
        self.result_matrices.get(index).map(|mat| {
            let slice_container = crate::SliceContainer::new(mat.data());
            SliceArray::new(BaseArray::new_with_stride(
                slice_container,
                mat.shape(),
                mat.stride(),
            ))
        })
    }

    fn result_matrix_mut(&mut self, index: usize) -> Option<SliceArrayMut<'_, Self::Item, 2>> {
        self.result_matrices.get_mut(index).map(|mat| {
            let shape = mat.shape();
            let stride = mat.stride();
            let slice_container = crate::SliceContainerMut::new(mat.data_mut());
            SliceArrayMut::new(BaseArray::new_with_stride(slice_container, shape, stride))
        })
    }

    fn evaluate(&mut self) -> RlstResult<()> {
        for index in 0..self.number_of_matrices {
            let left_matrix = self.left_matrices[index].view();
            let right_matrix = self.right_matrices[index].view();
            let result_matrix = self.result_matrices[index].view_mut();
            result_matrix.mult_into(
                TransMode::NoTrans,
                TransMode::NoTrans,
                self.alpha,
                left_matrix,
                right_matrix,
                self.beta,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::dense::traits::DefaultIterator;
    use crate::dense::traits::MultIntoResize;

    #[test]
    pub fn test_batched_cpu_gemm() {
        let mut batched_matmul = DefaultCpuBatchedGemm::<f64>::new((2, 3), (3, 5), 2, 1.0, 0.0);

        batched_matmul
            .left_matrix_mut(0)
            .unwrap()
            .fill_from_seed_equally_distributed(0);
        batched_matmul
            .left_matrix_mut(1)
            .unwrap()
            .fill_from_seed_equally_distributed(1);

        batched_matmul
            .right_matrix_mut(0)
            .unwrap()
            .fill_from_seed_equally_distributed(2);
        batched_matmul
            .right_matrix_mut(1)
            .unwrap()
            .fill_from_seed_equally_distributed(3);

        batched_matmul.evaluate().unwrap();

        for index in 0..2 {
            let expected = crate::dense::array::empty_array().simple_mult_into_resize(
                batched_matmul.left_matrix(index).unwrap(),
                batched_matmul.right_matrix(index).unwrap(),
            );

            crate::assert_array_relative_eq!(
                expected,
                batched_matmul.result_matrix(index).unwrap(),
                1E-12
            );
        }
    }
}
