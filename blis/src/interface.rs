pub mod gemm;
pub mod types;

use cauchy::Scalar;

/// Compute expected size of a data slice from stride and shape.
pub fn get_expected_data_size(stride: [usize; 2], shape: [usize; 2]) -> usize {
    if shape[0] == 0 || shape[1] == 0 {
        return 0;
    }

    1 + (shape[0] - 1) * stride[0] + (shape[1] - 1) * stride[1]
}

/// Panic if expected data size is not identical to actual data size.
pub fn assert_data_size<T: Scalar>(data: &[T], stride: [usize; 2], shape: [usize; 2]) {
    let expected = get_expected_data_size(stride, shape);

    assert_eq!(
        expected,
        data.len(),
        "Wrong size for data slice. Actual size {}. Expected size {}.",
        data.len(),
        expected
    );
}
