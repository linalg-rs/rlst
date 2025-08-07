//! Matrix inverse.
//!
//!
use super::interface::getrf::Getrf;
use super::interface::getri::Getri;
use crate::base_types::RlstResult;
use crate::dense::array::{Array, DynArray};
use crate::traits::base_operations::Shape;
use crate::traits::linalg::decompositions::Inverse;
use crate::UnsafeRandom1DAccessByValue;

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

        Item::getrf(m, n, out.data_mut(), m, &mut ipiv)?;
        Item::getri(n, out.data_mut(), m, &ipiv)?;

        Ok(out)
    }
}
