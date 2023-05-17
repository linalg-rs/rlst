//! Useful library tools.

use crate::traits::properties::Shape;
use crate::traits::RandomAccessByValue;
use crate::types::*;
use rand::prelude::*;
use rand_distr::Distribution;

/// This trait implements a simple convenient function to return random scalars
/// from a given random number generator and distribution. For complex types the
/// generator and distribution are separately applied to obtain the real and imaginary
/// part of the random number.
pub trait RandScalar: Scalar {
    /// Returns a random number from a given random number generator `rng` and associated
    /// distribution `dist`.
    fn random_scalar<R: Rng, D: Distribution<<Self as Scalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self;
}

impl RandScalar for f32 {
    fn random_scalar<R: Rng, D: Distribution<<Self as Scalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self {
        dist.sample(rng)
    }
}

impl RandScalar for f64 {
    fn random_scalar<R: Rng, D: Distribution<<Self as Scalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self {
        dist.sample(rng)
    }
}

impl RandScalar for c32 {
    fn random_scalar<R: Rng, D: Distribution<<Self as Scalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self {
        c32::new(dist.sample(rng), dist.sample(rng))
    }
}

impl RandScalar for c64 {
    fn random_scalar<R: Rng, D: Distribution<<Self as Scalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self {
        c64::new(dist.sample(rng), dist.sample(rng))
    }
}

pub struct MatDisplayWrapper<'a, T: Scalar, Mat: RandomAccessByValue<Item = T> + Shape>(&'a Mat);

pub trait ToMatDisplayWrapper<'a, T: Scalar, Mat: RandomAccessByValue<Item = T> + Shape> {
    fn to_stdout(&'a self) -> MatDisplayWrapper<'a, T, Mat>;
}

impl<'a, T: Scalar, Mat: RandomAccessByValue<Item = T> + Shape> ToMatDisplayWrapper<'a, T, Mat>
    for Mat
{
    fn to_stdout(&'a self) -> MatDisplayWrapper<'a, T, Mat> {
        MatDisplayWrapper(self)
    }
}

impl<'a, T: Scalar, Mat: RandomAccessByValue<Item = T> + Shape> std::fmt::Display
    for MatDisplayWrapper<'a, T, Mat>
where
    f64: From<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let shape = self.0.shape();
        let mut content_str = String::new();

        // For alignment in columns, must satisfy: precision + exp_padding < width - 4
        let print_width = 10;
        let mantissa_precision = 2; // number of digits after decimal point
        let exponent_padding = 2; // number of digits (excluding the sign) in the exponent
        let num_outer_spaces = print_width - mantissa_precision - exponent_padding - 4;
        // let mut mytype = String::new();
        for row in 0..shape.0 {
            content_str += "│";
            for col in 0..shape.1 {
                #[allow(clippy::unwrap_used)]
                let x_ij = f64::from(self.0.get_value(row, col).unwrap());

                content_str += &format!(
                    " {}",
                    fmt_f64(x_ij, print_width, mantissa_precision, exponent_padding)
                );
            }
            content_str += &" ".repeat(num_outer_spaces);
            content_str += "│\n";
        }

        let top_str = format!(
            "\n┌{}┐\n",
            " ".repeat((shape.1 * (10 + 1)) + num_outer_spaces)
        );
        let btm_str = format!(
            "└{}┘\n",
            " ".repeat((shape.1 * (10 + 1)) + num_outer_spaces)
        );
        write!(f, "{top_str}{content_str}{btm_str}")
    }
}

impl<'a, T: Scalar, Mat: RandomAccessByValue<Item = T> + Shape> std::fmt::Debug
    for MatDisplayWrapper<'a, T, Mat>
where
    f64: From<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let shape = self.0.shape();
        let mut content_str = String::new();

        // For alignment in columns, must satisfy: precision + exp_padding < width - 4
        let print_width = 10;
        let mantissa_precision = 2; // number of digits after decimal point
        let exponent_padding = 2; // number of digits (excluding the sign) in the exponent
        let num_outer_spaces = print_width - mantissa_precision - exponent_padding - 4;
        // let mut mytype = String::new();
        for row in 0..shape.0 {
            content_str += "│";
            for col in 0..shape.1 {
                #[allow(clippy::unwrap_used)]
                let x_ij = f64::from(self.0.get_value(row, col).unwrap());

                content_str += &format!(
                    " {}",
                    fmt_f64(x_ij, print_width, mantissa_precision, exponent_padding)
                );
            }
            content_str += &" ".repeat(num_outer_spaces);
            content_str += "│\n";
        }

        let top_str = format!(
            "\n┌{}┐\n",
            " ".repeat((shape.1 * (10 + 1)) + num_outer_spaces)
        );
        let btm_str = format!(
            "└{}┘\n",
            " ".repeat((shape.1 * (10 + 1)) + num_outer_spaces)
        );
        write!(f, "{top_str}{content_str}{btm_str}")
    }
}

// https://stackoverflow.com/a/65266882
fn fmt_f64(num: f64, width: usize, precision: usize, exp_pad: usize) -> String {
    let mut num = format!("{:.precision$e}", num, precision = precision);
    // Safe to `unwrap` as `num` is guaranteed to contain `'e'`
    #[allow(clippy::unwrap_used)]
    let exp = num.split_off(num.find('e').unwrap());

    let (sign, exp) = exp
        .strip_prefix("e-")
        .map_or_else(|| ('+', &exp[1..]), |stripped| ('-', stripped));

    num.push_str(&format!("e{}{:0>pad$}", sign, exp, pad = exp_pad));

    format!("{:>width$}", num, width = width)
}
