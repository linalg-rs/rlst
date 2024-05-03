// //! Implementation of Batched Gemm using Metal
//
// use super::interface::{
//     MetalDevice, MpsDataType, MpsMatrix, MpsMatrixDescriptor, MpsMatrixMultiplication,
// };
// use crate::dense::batched_gemm::BatchedGemm;
// use crate::external::metal::interface::ResourceOptions;
// use crate::AutoReleasePool;
// use crate::RlstResult;
// use crate::SliceArrayMut;
// use crate::{BaseArray, SliceArray};
//
// /// Batched matrix multiplication using Apple Metal.
// pub struct MetalBatchedGemm {
//     left_matrices: MpsMatrix,
//     right_matrices: MpsMatrix,
//     result_matrices: MpsMatrix,
//     mps_matmat: MpsMatrixMultiplication,
//     number_of_matrices: usize,
//     device: MetalDevice,
// }
//
// impl MetalBatchedGemm {
//     /// Initialize a new Metal batched Matrix Multiplication.
//     pub fn new(
//         left_dim: (usize, usize),
//         right_dim: (usize, usize),
//         number_of_matrices: usize,
//         alpha: f64,
//         beta: f64,
//     ) -> Self {
//         let device = MetalDevice::from_default();
//
//         let left_row_bytes =
//             MpsMatrixDescriptor::row_bytes_from_columns(left_dim.1, MpsDataType::F32);
//         let right_row_bytes =
//             MpsMatrixDescriptor::row_bytes_from_columns(right_dim.1, MpsDataType::F32);
//         let result_row_bytes =
//             MpsMatrixDescriptor::row_bytes_from_columns(right_dim.1, MpsDataType::F32);
//
//         assert_eq!(left_dim.1, right_dim.0);
//
//         let left_matrix_bytes = left_dim.0 * left_row_bytes;
//         let right_matrix_bytes = right_dim.0 * right_row_bytes;
//         let result_matrix_bytes = left_dim.0 * result_row_bytes;
//
//         let mps_matmat = MpsMatrixMultiplication::new(
//             &device,
//             false,
//             false,
//             left_dim.0,
//             right_dim.1,
//             left_dim.1,
//             alpha,
//             beta,
//         );
//
//         let left_buffer = device.new_buffer(
//             left_matrix_bytes * number_of_matrices,
//             ResourceOptions::HazardTrackingModeUntracked as u32,
//         );
//         let right_buffer = device.new_buffer(
//             right_matrix_bytes * number_of_matrices,
//             ResourceOptions::HazardTrackingModeUntracked as u32,
//         );
//         let result_buffer = device.new_buffer(
//             result_matrix_bytes * number_of_matrices,
//             ResourceOptions::HazardTrackingModeUntracked as u32,
//         );
//
//         let left_matrix_desc = MpsMatrixDescriptor::new(
//             left_dim.0,
//             left_dim.1,
//             number_of_matrices,
//             left_row_bytes,
//             left_matrix_bytes,
//             MpsDataType::F32,
//         );
//
//         let right_matrix_desc = MpsMatrixDescriptor::new(
//             right_dim.0,
//             right_dim.1,
//             number_of_matrices,
//             right_row_bytes,
//             right_matrix_bytes,
//             MpsDataType::F32,
//         );
//
//         let result_matrix_desc = MpsMatrixDescriptor::new(
//             left_dim.0,
//             right_dim.1,
//             number_of_matrices,
//             result_row_bytes,
//             result_matrix_bytes,
//             MpsDataType::F32,
//         );
//
//         let left_matrices = MpsMatrix::new(left_buffer, left_matrix_desc);
//         let right_matrices = MpsMatrix::new(right_buffer, right_matrix_desc);
//         let result_matrices = MpsMatrix::new(result_buffer, result_matrix_desc);
//
//         Self {
//             left_matrices,
//             right_matrices,
//             result_matrices,
//             mps_matmat,
//             number_of_matrices,
//             device,
//         }
//     }
// }
//
// impl BatchedGemm for MetalBatchedGemm {
//     type Item = f32;
//
//     fn left_matrix(&self, index: usize) -> Option<SliceArray<'_, Self::Item, 2>> {
//         if index < self.number_of_matrices {
//             let desc = self.left_matrices.descriptor();
//             let contents = self.left_matrices.contents::<f32>();
//             let matrix_elements = desc.matrix_bytes() / std::mem::size_of::<Self::Item>();
//             let row_stride = desc.row_bytes() / std::mem::size_of::<Self::Item>();
//             let slice = &contents[matrix_elements * index..matrix_elements * (1 + index)];
//             let container = crate::SliceContainer::new(slice);
//             let shape = [desc.rows(), desc.columns()];
//             let stride = [row_stride, 1];
//             Some(SliceArray::new(BaseArray::new_with_stride(
//                 container, shape, stride,
//             )))
//         } else {
//             None
//         }
//     }
//
//     fn left_matrix_mut(&mut self, index: usize) -> Option<SliceArrayMut<'_, Self::Item, 2>> {
//         if index < self.number_of_matrices {
//             let desc = self.left_matrices.descriptor();
//             let shape = [desc.rows(), desc.columns()];
//             let row_stride = desc.row_bytes() / std::mem::size_of::<Self::Item>();
//             let stride = [row_stride, 1];
//             let matrix_elements = desc.matrix_bytes() / std::mem::size_of::<Self::Item>();
//             let contents = self.left_matrices.contents_mut::<f32>();
//             let slice = &mut contents[matrix_elements * index..matrix_elements * (1 + index)];
//             let container = crate::SliceContainerMut::new(slice);
//             Some(SliceArrayMut::new(BaseArray::new_with_stride(
//                 container, shape, stride,
//             )))
//         } else {
//             None
//         }
//     }
//
//     fn right_matrix(&self, index: usize) -> Option<SliceArray<'_, Self::Item, 2>> {
//         if index < self.number_of_matrices {
//             let desc = self.right_matrices.descriptor();
//             let contents = self.right_matrices.contents::<f32>();
//             let matrix_elements = desc.matrix_bytes() / std::mem::size_of::<Self::Item>();
//             let row_stride = desc.row_bytes() / std::mem::size_of::<Self::Item>();
//             let slice = &contents[matrix_elements * index..matrix_elements * (1 + index)];
//             let container = crate::SliceContainer::new(slice);
//             let shape = [desc.rows(), desc.columns()];
//             let stride = [row_stride, 1];
//             Some(SliceArray::new(BaseArray::new_with_stride(
//                 container, shape, stride,
//             )))
//         } else {
//             None
//         }
//     }
//
//     fn right_matrix_mut(&mut self, index: usize) -> Option<SliceArrayMut<'_, Self::Item, 2>> {
//         if index < self.number_of_matrices {
//             let desc = self.right_matrices.descriptor();
//             let shape = [desc.rows(), desc.columns()];
//             let row_stride = desc.row_bytes() / std::mem::size_of::<Self::Item>();
//             let stride = [row_stride, 1];
//             let matrix_elements = desc.matrix_bytes() / std::mem::size_of::<Self::Item>();
//             let contents = self.right_matrices.contents_mut::<f32>();
//             let slice = &mut contents[matrix_elements * index..matrix_elements * (1 + index)];
//             let container = crate::SliceContainerMut::new(slice);
//             Some(SliceArrayMut::new(BaseArray::new_with_stride(
//                 container, shape, stride,
//             )))
//         } else {
//             None
//         }
//     }
//     fn result_matrix(&self, index: usize) -> Option<SliceArray<'_, Self::Item, 2>> {
//         if index < self.number_of_matrices {
//             let desc = self.result_matrices.descriptor();
//             let contents = self.result_matrices.contents::<f32>();
//             let matrix_elements = desc.matrix_bytes() / std::mem::size_of::<Self::Item>();
//             let row_stride = desc.row_bytes() / std::mem::size_of::<Self::Item>();
//             let slice = &contents[matrix_elements * index..matrix_elements * (1 + index)];
//             let container = crate::SliceContainer::new(slice);
//             let shape = [desc.rows(), desc.columns()];
//             let stride = [row_stride, 1];
//             Some(SliceArray::new(BaseArray::new_with_stride(
//                 container, shape, stride,
//             )))
//         } else {
//             None
//         }
//     }
//
//     fn result_matrix_mut(&mut self, index: usize) -> Option<SliceArrayMut<'_, Self::Item, 2>> {
//         if index < self.number_of_matrices {
//             let desc = self.result_matrices.descriptor();
//             let shape = [desc.rows(), desc.columns()];
//             let row_stride = desc.row_bytes() / std::mem::size_of::<Self::Item>();
//             let stride = [row_stride, 1];
//             let matrix_elements = desc.matrix_bytes() / std::mem::size_of::<Self::Item>();
//             let contents = self.result_matrices.contents_mut::<f32>();
//             let slice = &mut contents[matrix_elements * index..matrix_elements * (1 + index)];
//             let container = crate::SliceContainerMut::new(slice);
//             Some(SliceArrayMut::new(BaseArray::new_with_stride(
//                 container, shape, stride,
//             )))
//         } else {
//             None
//         }
//     }
//
//     /// Evaluate the batched matrix product.
//     fn evaluate(&mut self) -> RlstResult<()> {
//         AutoReleasePool::execute(|| {
//             let command_queue = self.device.command_queue();
//             let mut command_buffer = command_queue.command_buffer();
//             self.mps_matmat.encode_to_command_buffer(
//                 &mut command_buffer,
//                 &self.left_matrices,
//                 &self.right_matrices,
//                 &mut self.result_matrices,
//             );
//             command_buffer.commit();
//             command_buffer.wait_until_completed();
//         });
//
//         Ok(())
//     }
// }
//
// #[cfg(test)]
// mod test {
//
//     use super::*;
//     use crate::{
//         empty_array, external::metal::AutoReleasePool, rlst_dynamic_array2, MultIntoResize,
//     };
//
//     use crate::prelude::*;
//
//     #[test]
//     fn test_batched_metal_gemm() {
//         AutoReleasePool::execute(|| {
//             let number_of_matrices = 3;
//             let mut batched_gemm =
//                 MetalBatchedGemm::new((3, 5), (5, 4), number_of_matrices, 1.0, 0.0);
//
//             for index in 0..number_of_matrices {
//                 let mut left_matrix = batched_gemm.left_matrix_mut(index).unwrap();
//                 left_matrix.fill_from_seed_equally_distributed(0);
//                 let mut right_matrix = batched_gemm.right_matrix_mut(index).unwrap();
//                 right_matrix.fill_from_seed_equally_distributed(1);
//             }
//
//             batched_gemm.evaluate().unwrap();
//
//             for index in 0..number_of_matrices {
//                 let mut expected = empty_array();
//                 let mut left_matrix = rlst_dynamic_array2!(f32, [3, 5]);
//                 let mut right_matrix = rlst_dynamic_array2!(f32, [5, 4]);
//                 left_matrix.fill_from(batched_gemm.left_matrix(index).unwrap());
//                 right_matrix.fill_from(batched_gemm.right_matrix(index).unwrap());
//                 expected
//                     .view_mut()
//                     .simple_mult_into_resize(left_matrix, right_matrix);
//                 crate::assert_array_relative_eq!(
//                     batched_gemm.result_matrix(index).unwrap(),
//                     expected,
//                     1E-6
//                 );
//             }
//         });
//     }
// }
