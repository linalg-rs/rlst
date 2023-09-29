//! Column major layout with arbitrary stride.
//!
//! This layout uses an arbitrary stride in memory with 1d indexing
//! in column major order. For further information on memory layouts
//! see [crate::traits::layout].

/// Compute stride from a shape
pub fn stride_from_shape<const NDIM: usize>(shape: [usize; NDIM]) -> [usize; NDIM] {
    let mut output = [0; NDIM];

    let mut state = 1;
    for (elem, s) in output.iter_mut().zip(shape.iter()) {
        *elem = state;
        state *= s;
    }
    output
}

/// Return true if `multi_index` in bounds with respect to `shape`.
pub fn check_multi_index_in_bounds<const N: usize>(
    multi_index: [usize; N],
    shape: [usize; N],
) -> bool {
    for (ind, s) in multi_index.iter().zip(shape.iter()) {
        if ind >= s {
            return false;
        }
    }
    true
}

/// Convert an n-d index into a 1d index.
#[inline]
pub fn convert_nd_raw<const NDIM: usize>(
    multi_index: [usize; NDIM],
    stride: [usize; NDIM],
) -> usize {
    let mut acc = 0;

    for ind in 0..NDIM {
        acc += multi_index[ind] * stride[ind];
    }

    acc
}

/// Convert a 1d index into a nd index.
#[inline]
pub fn convert_1d_nd_from_shape<const NDIM: usize>(
    mut index: usize,
    shape: [usize; NDIM],
) -> [usize; NDIM] {
    let mut res = [0; NDIM];
    debug_assert!(index < shape.iter().product());
    for ind in 0..NDIM {
        res[ind] = index % shape[ind];
        index /= shape[ind];
    }
    res
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_convert_1d_nd() {
        let multi_index: [usize; 3] = [2, 3, 7];
        let shape: [usize; 3] = [3, 4, 8];
        let stride = stride_from_shape(shape);

        let index_1d = convert_nd_raw(multi_index, stride);
        let actual_nd = convert_1d_nd_from_shape(index_1d, shape);

        println!("{}, {:#?}", index_1d, actual_nd);

        for (&expected, actual) in multi_index.iter().zip(actual_nd) {
            assert_eq!(expected, actual)
        }
    }
}
