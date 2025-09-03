//! A collection of macros for Rlst.

/// Dot product between compatible arrays.
///
/// It always allocates a new array for the result.
/// Call as `res = dot!(a, b, c)` to multiply the matrices
/// `a`, `b,`, and `c`. The product of arbitrary numbers of matrices
/// is supported. For more than two arguments the product is evaluated
/// from back to front.
#[macro_export]
macro_rules! dot {
    ($a:expr, $b:expr) => {
        $a.dot(&$b)
    };

    ($a:expr, $b:expr, $($c:expr),+) => {{
        dot!($a, dot!($b, $($c),+))
    }};

}

/// Create a new n-dimensional diagonal array from a list of diagonal entries.
///
/// Call with `diag!(d)` where `d` is a one-dimensional array to obtain a two-dimensional array with `d` on the
/// diagonal. For n-dimensional diagonal arrays, call with `diag!(d, NDIM)` where `NDIM` is the
/// number of dimensions and `d` is a list of diagonal entries.
#[macro_export]
macro_rules! diag {
    ($d:expr) => {
        diag!($d, 2)
    };

    ($d:expr, $ndim:literal) => {{
        use itertools::izip;

        let mut diag_array =
            $crate::dense::array::DynArray::<_, $ndim>::from_shape([$d.len(); $ndim]);
        izip!(diag_array.diag_iter_mut(), $d.iter_value()).for_each(|(diag_entry, entry)| {
            *diag_entry = entry;
        });
        diag_array
    }};
}

#[cfg(test)]
mod test {

    use rand::SeedableRng;

    use crate::{dense::array::DynArray, empty_array, MultIntoResize};

    #[test]
    fn test_dot() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let mut arr1 = DynArray::<f64, 2>::from_shape([2, 3]);
        let mut arr2 = DynArray::<f64, 2>::from_shape([3, 4]);
        let mut arr3 = DynArray::<f64, 2>::from_shape([4, 5]);
        arr1.fill_from_standard_normal(&mut rng);
        arr2.fill_from_standard_normal(&mut rng);
        arr3.fill_from_standard_normal(&mut rng);

        let expected = empty_array().simple_mult_into_resize(
            empty_array().simple_mult_into_resize(arr1.r(), arr2.r()),
            arr3.r(),
        );

        let actual = crate::dot!(arr1.r(), arr2.r(), arr3.r());

        crate::assert_array_relative_eq!(expected, actual, 1E-10);
    }

    #[test]
    fn test_diag() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let mut d = DynArray::<f64, 1>::from_shape([2]);
        d.fill_from_equally_distributed(&mut rng);

        let actual = diag!(d);
        let mut expected = DynArray::<f64, 2>::from_shape([2, 2]);

        expected[[0, 0]] = d[[0]];
        expected[[1, 1]] = d[[1]];

        crate::assert_array_relative_eq!(actual, expected, 1E-10);
    }
}
