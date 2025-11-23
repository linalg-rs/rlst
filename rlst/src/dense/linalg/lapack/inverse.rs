//! Matrix inverse.
//!
//!
use super::interface::getrf::Getrf;
use super::interface::getri::Getri;
use crate::UnsafeRandom1DAccessByValue;
use crate::base_types::RlstResult;
use crate::dense::array::{Array, DynArray};
use crate::traits::base_operations::Shape;
use crate::traits::linalg::decompositions::Inverse;

impl<Item, ArrayImpl> Inverse for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    Item: Copy + Default + Getri + Getrf,
{
    type Output = DynArray<Item, 2>;

    fn inverse(&self) -> RlstResult<DynArray<Item, 2>> {
        let m = self.shape()[0];
        let n = self.shape()[1];
        assert_eq!(m, n, "Matrix must be square for inversion.");
        let mut out = DynArray::new_from(self);
        let mut ipiv = vec![0_i32; m];

        Item::getrf(m, n, out.data_mut().unwrap(), m, &mut ipiv)?;
        Item::getri(n, out.data_mut().unwrap(), m, &ipiv)?;

        Ok(out)
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::MultIntoResize;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::empty_array;
    use paste::paste;

    macro_rules! impl_inverse_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {
            #[test]
            fn [<test_inverse_$scalar>]() {
                let n = 10;

                let mut a = DynArray::<$scalar, 2>::from_shape([n, n]);
                a.fill_from_seed_equally_distributed(0);
                let inv = a.inverse().unwrap();

                let mut ident = DynArray::<$scalar, 2>::from_shape([n, n]);
                ident.set_identity();

                let actual = empty_array::<$scalar, 2>().simple_mult_into_resize(inv.r(), a.r());
                crate::assert_array_abs_diff_eq!(actual, ident, $tol);
            }
            }
        };
    }

    impl_inverse_tests!(f32, 1E-5);
    impl_inverse_tests!(f64, 1E-10);
    impl_inverse_tests!(c32, 1E-4);
    impl_inverse_tests!(c64, 1E-10);
}
