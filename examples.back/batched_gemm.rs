//! Example of batched matrix multiplication

fn main() {
    //     #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    //     let pool = AutoReleasePool::new();
    //
    //     let left_dim = (53, 77);
    //     let right_dim = (77, 69);
    //
    //     let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    //
    //     let number_of_matrices = 1;
    //
    //     #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    //     let mut batched_gemm = MetalBatchedGemm::new(left_dim, right_dim, number_of_matrices, 1.0, 0.0);
    //
    //     #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    //     let mut batched_gemm =
    //         DefaultCpuBatchedGemm::new(left_dim, right_dim, number_of_matrices, 1.0, 0.0);
    //
    //     for index in 0..number_of_matrices {
    //         let mut left_matrix = batched_gemm.left_matrix_mut(index).unwrap();
    //         left_matrix.fill_from_equally_distributed(&mut rng);
    //         let mut right_matrix = batched_gemm.right_matrix_mut(index).unwrap();
    //         right_matrix.fill_from_equally_distributed(&mut rng);
    //     }
    //
    //     batched_gemm.evaluate().unwrap();
    //
    //     // Check that the result is correct.
    //     for index in 0..number_of_matrices {
    //         let mut left_matrix = rlst_dynamic_array2!(f32, [left_dim.0, left_dim.1]);
    //         let mut right_matrix = rlst_dynamic_array2!(f32, [right_dim.0, right_dim.1]);
    //
    //         left_matrix.fill_from(batched_gemm.left_matrix(index).unwrap());
    //         right_matrix.fill_from(batched_gemm.right_matrix(index).unwrap());
    //
    //         let mut expected = empty_array();
    //
    //         expected
    //             .r_mut()
    //             .simple_mult_into_resize(left_matrix.r(), right_matrix.r());
    //
    //         assert_array_relative_eq!(expected, batched_gemm.result_matrix(index).unwrap(), 1E-5);
    //     }
    //
    //     #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    //     pool.drain();
}
