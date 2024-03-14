//! Default layout for dense arrays.
//!
//! The default layout is a column-major format such that in an
//! index `[a0, a1, a2, ...]` the left-most axis is stored consecutively in memory.
//!

/// Compute stride from a shape.
///
/// Given an array shape, compute the corresponding stride assuming column-major ordering.
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

/// Convert a multi-index into a 1d index.
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

/// Convert a 1d index into a multi-index.
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
