//! Useful library tools.

use rand::prelude::*;
use rand_distr::Distribution;

use crate::base_types::{c32, c64};
use crate::traits::accessors::RandomAccessByValue;
use crate::traits::base_operations::Shape;
use crate::traits::rlst_num::{RandScalar, RlstScalar};

impl RandScalar for f32 {
    fn random_scalar<R: Rng, D: Distribution<<Self as RlstScalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self {
        dist.sample(rng)
    }
}

impl RandScalar for f64 {
    fn random_scalar<R: Rng, D: Distribution<<Self as RlstScalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self {
        dist.sample(rng)
    }
}

impl RandScalar for c32 {
    fn random_scalar<R: Rng, D: Distribution<<Self as RlstScalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self {
        Self::new(dist.sample(rng), dist.sample(rng))
    }
}

impl RandScalar for c64 {
    fn random_scalar<R: Rng, D: Distribution<<Self as RlstScalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self {
        Self::new(dist.sample(rng), dist.sample(rng))
    }
}

/// Assert that the absolute difference of two matrices is small enough
#[macro_export]
macro_rules! assert_matrix_abs_diff_eq {
    ($expected_matrix:expr, $actual_matrix:expr, $epsilon:expr) => {{
        use approx::assert_abs_diff_eq;
        assert_eq!($expected_matrix.shape(), $actual_matrix.shape());
        for row in 0..$expected_matrix.shape().0 {
            for col in 0..$expected_matrix.shape().1 {
                assert_abs_diff_eq!(
                    $actual_matrix[[row, col]],
                    $expected_matrix[[row, col]],
                    epsilon = $epsilon
                );
            }
        }
    }};
}

/// Assert that the relative difference of two matrices is small enough
#[macro_export]
macro_rules! assert_matrix_relative_eq {
    ($expected_matrix:expr, $actual_matrix:expr, $epsilon:expr) => {{
        use approx::assert_relative_eq;
        assert_eq!($expected_matrix.shape(), $actual_matrix.shape());
        for row in 0..$expected_matrix.shape().0 {
            for col in 0..$expected_matrix.shape().1 {
                assert_relative_eq!(
                    $actual_matrix[[row, col]],
                    $expected_matrix[[row, col]],
                    max_relative = $epsilon
                );
            }
        }
    }};
}

/// Assert that the absolute difference of two arrays is small enough
#[macro_export]
macro_rules! assert_array_abs_diff_eq {
    ($expected_array:expr, $actual_array:expr, $epsilon:expr) => {{
        use approx::assert_abs_diff_eq;
        use $crate::traits::base_operations::Shape;
        use $crate::traits::iterators::ArrayIteratorByValue;
        assert_eq!($expected_array.shape(), $actual_array.shape());
        for (actual, expected) in $actual_array.iter_value().zip($expected_array.iter_value()) {
            assert_abs_diff_eq!(actual, expected, epsilon = $epsilon);
        }
    }};
}

/// Assert that the relative difference of two arrays is small enough
#[macro_export]
macro_rules! assert_array_relative_eq {
    ($expected_array:expr, $actual_array:expr, $epsilon:expr) => {{
        use approx::assert_relative_eq;
        use $crate::traits::base_operations::Shape;
        use $crate::traits::iterators::ArrayIteratorByValue;
        assert_eq!($expected_array.shape(), $actual_array.shape());
        for (actual, expected) in $actual_array.iter_value().zip($expected_array.iter_value()) {
            assert_relative_eq!(actual, expected, max_relative = $epsilon);
        }
    }};
}

/// TODO: document
#[macro_export]
macro_rules! assert_matrix_ulps_eq {
    ($expected_matrix:expr, $actual_matrix:expr, $ulps:expr) => {{
        use approx::assert_abs_ulps_eq;
        assert_eq!($expected_matrix.shape(), $actual_matrix.shape());
        for row in 0..$expected_matrix.shape().0 {
            for col in 0..$expected_matrix.shape().1 {
                assert_abs_ulps_eq!(
                    $actual_matrix[[row, col]],
                    $expected_matrix[[row, col]],
                    max_ulps = $ulps
                );
            }
        }
    }};
}

/// Pretty printing
pub trait PrettyPrint<T: RlstScalar> {
    /// Pretty printing
    fn pretty_print(&self);
    /// Pretty printing with dimension
    fn pretty_print_with_dimension(&self, rows: usize, cols: usize);
    /// Pretty printing with advanced options
    fn pretty_print_advanced(
        &self,
        rows: usize,
        cols: usize,
        print_width: usize,
        mantissa: usize,
        exponent: usize,
    );
}

macro_rules! pretty_print_impl {
    ($scalar:ty, $fmtfun:ident) => {
        impl<Mat: RandomAccessByValue<2, Item = $scalar> + Shape<2>> PrettyPrint<$scalar> for Mat {
            fn pretty_print(&self) {
                self.pretty_print_advanced(10, 10, 11, 3, 2)
            }

            fn pretty_print_with_dimension(&self, rows: usize, cols: usize) {
                self.pretty_print_advanced(rows, cols, 11, 3, 2)
            }

            fn pretty_print_advanced(
                &self,
                rows: usize,
                cols: usize,
                print_width: usize,
                mantissa: usize,
                exponent: usize,
            ) {
                let shape = (
                    std::cmp::min(self.shape()[0], rows),
                    std::cmp::min(self.shape()[1], cols),
                );
                let mut content_str = String::new();

                // For alignment in columns, must satisfy: 4 + mantissa + exponent < print_width
                let print_width = std::cmp::max(print_width, 5 + mantissa + exponent);

                let num_outer_spaces = print_width - mantissa - exponent - 4;

                let mut x_ij = self.get_value([0, 0]).unwrap();
                for row in 0..shape.0 {
                    content_str += "│";
                    for col in 0..shape.1 {
                        x_ij = self.get_value([row, col]).unwrap();

                        content_str += &format!(
                            " {}",
                            $fmtfun::<$scalar>(x_ij, print_width, mantissa, exponent)
                        );
                    }
                    content_str += &" ".repeat(num_outer_spaces);
                    content_str += "│\n";
                }

                let colwidth = $fmtfun::<$scalar>(x_ij, print_width, mantissa, exponent)
                    .chars()
                    .count();
                let top_str = format!(
                    "\n┌{}┐\n",
                    " ".repeat((shape.1 * (colwidth + 1)) + num_outer_spaces)
                );
                let btm_str = format!(
                    "└{}┘\n",
                    " ".repeat((shape.1 * (colwidth + 1)) + num_outer_spaces)
                );
                println!(
                    "Printing the upper left {} x {} block of a matrix with dimensions {} x {}.",
                    shape.0,
                    shape.1,
                    self.shape()[0],
                    self.shape()[1]
                );
                println!("{top_str}{content_str}{btm_str}");
            }
        }
    };
}

pretty_print_impl!(f64, fmt_real);
pretty_print_impl!(f32, fmt_real);
pretty_print_impl!(c32, fmt_complex);
pretty_print_impl!(c64, fmt_complex);

// https://stackoverflow.com/a/65266882
fn fmt_real<T: RlstScalar>(num: T::Real, width: usize, precision: usize, exp_pad: usize) -> String {
    let mut num = format!("{num:.precision$e}");
    // Safe to `unwrap` as `num` is guaranteed to contain `'e'`
    let exp = num.split_off(num.find('e').unwrap());

    let (sign, exp) = exp
        .strip_prefix("e-")
        .map_or_else(|| ('+', &exp[1..]), |stripped| ('-', stripped));

    num.push_str(&format!("e{sign}{exp:0>exp_pad$}"));

    format!("{num:>width$}")
}

fn fmt_complex<T: RlstScalar>(num: T, width: usize, precision: usize, exp_pad: usize) -> String {
    let sign = if num.im() < <T::Real as num::Zero>::zero() {
        "-"
    } else {
        "+"
    };
    let mut printstr = format!(
        "{}{}",
        fmt_real::<T::Real>(num.re(), width, precision, exp_pad),
        fmt_real::<T::Real>(num.im().abs(), width, precision, exp_pad),
    );
    let num_blanks = width - precision - exp_pad - 4;
    if num_blanks % 2 == 0 {
        printstr = format!(
            "{} {}𝑖",
            fmt_real::<T::Real>(num.re(), width, precision, exp_pad),
            fmt_real::<T::Real>(num.im().abs(), width, precision, exp_pad),
        );
    }
    let sign_position = width + (num_blanks / 2);

    printstr.replace_range(sign_position..=sign_position, sign);
    printstr
}
