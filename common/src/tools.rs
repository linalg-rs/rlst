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

macro_rules! dense_mat_display {
    ($matdisplaywrapper:tt, $tomatdisplaywrapper:tt, $fmtfn:ident, $type:ty) => {
        #[derive(Debug)]
        #[allow(non_camel_case_types)]
        pub struct $matdisplaywrapper<'a, Mat: RandomAccessByValue<Item = $type> + Shape>(&'a Mat);

        #[allow(non_camel_case_types)]
        pub trait $tomatdisplaywrapper<'a, Mat: RandomAccessByValue<Item = $type> + Shape> {
            fn to_stdout(&'a self) -> $matdisplaywrapper<'a, Mat>;
        }

        impl<'a, Mat: RandomAccessByValue<Item = $type> + Shape> $tomatdisplaywrapper<'a, Mat>
            for Mat
        {
            fn to_stdout(&'a self) -> $matdisplaywrapper<'a, Mat> {
                $matdisplaywrapper(self)
            }
        }

        impl<'a, Mat: RandomAccessByValue<Item = $type> + Shape> std::fmt::Display
            for $matdisplaywrapper<'a, Mat>
        {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let shape = self.0.shape();
                let mut content_str = String::new();

                // For alignment in columns, must satisfy: precision + exp_padding < width - 4
                let print_width = 11;
                let mantissa_precision = 3; // number of digits after decimal point
                let exponent_padding = 2; // number of digits (excluding the sign) in the exponent
                let num_outer_spaces = print_width - mantissa_precision - exponent_padding - 4;

                let mut x_ij = self.0.get_value(0, 0).unwrap();
                for row in 0..shape.0 {
                    content_str += "│";
                    for col in 0..shape.1 {
                        x_ij = self.0.get_value(row, col).unwrap();

                        content_str += &format!(
                            " {}",
                            $fmtfn(x_ij, print_width, mantissa_precision, exponent_padding)
                        );
                    }
                    content_str += &" ".repeat(num_outer_spaces);
                    content_str += "│\n";
                }

                let colwidth = $fmtfn(x_ij, print_width, mantissa_precision, exponent_padding)
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
                write!(f, "{top_str}{content_str}{btm_str}")
            }
        }
    };
}

dense_mat_display!(MatDisplayWrapper_f64, ToMatDisplayWrapper_f64, fmt_f64, f64);
dense_mat_display!(MatDisplayWrapper_c64, ToMatDisplayWrapper_c64, fmt_c64, c64);

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

fn fmt_c64(num: c64, width: usize, precision: usize, exp_pad: usize) -> String {
    let sign = if num.im < 0.0 { "-" } else { "+" };
    let mut printstr = format!(
        "{}{}",
        fmt_f64(num.re, width, precision, exp_pad),
        fmt_f64(num.im.abs(), width, precision, exp_pad),
    );
    let num_blanks = width - precision - exp_pad - 4;
    if num_blanks % 2 == 0 {
        printstr = format!(
            "{} {}𝑖",
            fmt_f64(num.re, width, precision, exp_pad),
            fmt_f64(num.im.abs(), width, precision, exp_pad),
        );
    }
    let sign_position = width + (num_blanks / 2);

    printstr.replace_range(sign_position..=sign_position, sign);
    printstr
}
