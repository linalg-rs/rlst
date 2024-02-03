use crate::{space::*, AsApply, OperatorBase};
use rlst_blis::interface::gemm::Gemm;
use rlst_common::types::Scalar;
use rlst_dense::{
    array::Array,
    traits::{MultInto, RawAccess, Shape, Stride, UnsafeRandomAccessByValue},
};

use super::array_vector_space::ArrayVectorSpace;

pub struct DenseMatrixOperator<
    'a,
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
> {
    arr: Array<Item, ArrayImpl, 2>,
    domain: &'a ArrayVectorSpace<Item>,
    range: &'a ArrayVectorSpace<Item>,
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
    > std::fmt::Debug for DenseMatrixOperator<'a, Item, ArrayImpl>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseMatrixOperator")
            .field("arr", &self.arr)
            .finish()
    }
}

impl<
        'a,
        Item: Scalar + Gemm,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccess<Item = Item>
            + 'a,
    > DenseMatrixOperator<'a, Item, ArrayImpl>
{
    pub fn from(
        arr: Array<Item, ArrayImpl, 2>,
        domain: &'a ArrayVectorSpace<Item>,
        range: &'a ArrayVectorSpace<Item>,
    ) -> crate::operator::RlstOperator<'a, ArrayVectorSpace<Item>, ArrayVectorSpace<Item>> {
        let shape = arr.shape();
        assert_eq!(domain.dimension(), shape[1]);
        assert_eq!(range.dimension(), shape[0]);
        Box::new(Self { arr, domain, range })
    }
}

impl<
        'a,
        Item: Scalar + Gemm,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
    > OperatorBase for DenseMatrixOperator<'a, Item, ArrayImpl>
{
    type Domain = ArrayVectorSpace<Item>;
    type Range = ArrayVectorSpace<Item>;

    fn as_apply(&self) -> Option<&dyn crate::AsApply<Domain = Self::Domain, Range = Self::Range>> {
        Some(self)
    }

    fn has_apply(&self) -> bool {
        true
    }

    fn domain(&self) -> &Self::Domain {
        self.domain
    }

    fn range(&self) -> &Self::Range {
        self.range
    }
}

impl<
        'a,
        Item: Scalar + Gemm,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
    > AsApply for DenseMatrixOperator<'a, Item, ArrayImpl>
{
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> rlst_common::types::RlstResult<()> {
        use rlst_dense::array::mult_into::TransMode;
        y.view_mut().mult_into(
            TransMode::NoTrans,
            TransMode::NoTrans,
            alpha,
            self.arr.view(),
            x.view(),
            beta,
        );
        Ok(())
    }
}

#[cfg(test)]
mod test {

    use rlst_dense::rlst_dynamic_array2;

    use crate::implementation::array_vector_space::ArrayVectorSpace;
    use crate::Element;
    use crate::LinearSpace;

    use super::DenseMatrixOperator;

    #[test]
    fn test_mat() {
        let mut mat = rlst_dynamic_array2!(f64, [3, 4]);
        let domain = ArrayVectorSpace::new(4);
        let range = ArrayVectorSpace::new(3);
        mat.fill_from_seed_equally_distributed(0);

        let op = DenseMatrixOperator::from(mat, &domain, &range);
        let mut x = op.domain().zero();
        let mut y = op.range().zero();

        x.view_mut().fill_from_seed_equally_distributed(0);

        op.as_apply().unwrap().apply(1.0, &x, 0.0, &mut y).unwrap();
    }
}
