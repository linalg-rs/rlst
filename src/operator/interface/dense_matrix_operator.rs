//! Dense matrix operator
use crate::dense::types::RlstScalar;
use crate::dense::{
    array::Array,
    traits::{MultInto, RawAccess, Shape, Stride, UnsafeRandomAccessByValue},
};
use crate::{
    operator::space::{Element, IndexableSpace, LinearSpace},
    operator::AsApply,
    operator::OperatorBase,
};

use super::array_vector_space::ArrayVectorSpace;

/// Dense matrix operator
pub struct DenseMatrixOperator<
    'a,
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
> {
    arr: Array<Item, ArrayImpl, 2>,
    domain: &'a ArrayVectorSpace<Item>,
    range: &'a ArrayVectorSpace<Item>,
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
    > std::fmt::Debug for DenseMatrixOperator<'_, Item, ArrayImpl>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseMatrixOperator")
            .field("arr", &self.arr)
            .finish()
    }
}

impl<
        'a,
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccess<Item = Item>
            + 'a,
    > DenseMatrixOperator<'a, Item, ArrayImpl>
{
    /// Create a new dense matrix operator
    pub fn new(
        arr: Array<Item, ArrayImpl, 2>,
        domain: &'a ArrayVectorSpace<Item>,
        range: &'a ArrayVectorSpace<Item>,
    ) -> Self {
        let shape = arr.shape();
        assert_eq!(domain.dimension(), shape[1]);
        assert_eq!(range.dimension(), shape[0]);
        Self { arr, domain, range }
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
    > OperatorBase for DenseMatrixOperator<'_, Item, ArrayImpl>
{
    type Domain = ArrayVectorSpace<Item>;
    type Range = ArrayVectorSpace<Item>;

    fn domain(&self) -> &Self::Domain {
        self.domain
    }

    fn range(&self) -> &Self::Range {
        self.range
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
    > AsApply for DenseMatrixOperator<'_, Item, ArrayImpl>
{
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> crate::dense::types::RlstResult<()> {
        use crate::dense::array::mult_into::TransMode;
        y.view_mut().mult_into(
            TransMode::NoTrans,
            TransMode::NoTrans,
            alpha,
            self.arr.r(),
            x.view().r(),
            beta,
        );
        Ok(())
    }
}
