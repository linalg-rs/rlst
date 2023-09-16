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

/// Convert an n-d index into a 1d index.
#[inline]
pub fn convert_nd_1d<const NDIM: usize>(indices: [usize; NDIM], stride: [usize; NDIM]) -> usize {
    indices
        .iter()
        .zip(stride.iter())
        .fold(0, |acc, (ind, s)| acc + ind * s)
}

/// Convert a 1d index into a nd index.
#[inline]
pub fn convert_1d_nd<const NDIM: usize>(mut index: usize, stride: [usize; NDIM]) -> [usize; NDIM] {
    let mut res = [0; NDIM];

    for ind in (0..NDIM).rev() {
        res[ind] = index / stride[ind];
        index %= stride[ind];
    }

    res
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_convert_1d_nd() {
        let indices: [usize; 4] = [3, 7, 14, 5];
        let shape: [usize; 4] = [4, 15, 17, 6];
        let stride = stride_from_shape(shape);

        let index_1d = convert_nd_1d(indices, stride);
        let actual_nd = convert_1d_nd(index_1d, stride);

        for (&expected, actual) in indices.iter().zip(actual_nd) {
            assert_eq!(expected, actual)
        }
    }
}
