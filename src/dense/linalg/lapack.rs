//! Lapack interface for linear algebra operations.

pub mod inverse;
pub mod lu;

use inverse::LapackInverse;
use lu::{ComputedLu, LapackLu, LuDecomposition};

use crate::{
    dense::{
        array::Array,
        traits::{RawAccess, RawAccessMut, Shape, Stride},
    },
    BaseItem,
};

/// Return a triple (m, n, lda) for the Lapack interface.
pub fn lapack_dims<ArrayImpl>(arr: &Array<ArrayImpl, 2>) -> (i32, i32, i32)
where
    ArrayImpl: Shape<2> + Stride<2>,
{
    let shape = arr.shape();
    let stride = arr.stride();

    let m = shape[0] as i32;
    let n = shape[1] as i32;
    assert_eq!(
        stride[0], 1,
        "Incorrect stride for Lapack. Stride[0] is {} but expected 1.",
        stride[0]
    );

    let lda = arr.stride()[1] as i32;

    (m, n, lda)
}

/// A wrapper for LAPACK operations.
pub struct LapackWrapper<Item, ArrayImpl>
where
    ArrayImpl: BaseItem<Item = Item> + Shape<2> + Stride<2>,
{
    arr: Array<ArrayImpl, 2>,
    _marker: std::marker::PhantomData<Item>,
}

impl<Item, ArrayImpl> LapackWrapper<Item, ArrayImpl>
where
    ArrayImpl: BaseItem<Item = Item> + Shape<2> + Stride<2>,
{
    /// Create a new LAPACK wrapper from an array.
    pub fn new(arr: Array<ArrayImpl, 2>) -> Self {
        assert_eq!(
            arr.stride()[0],
            1,
            "Incorrect stride for Lapack. Stride[0] is {} but expected 1.",
            arr.stride()[0]
        );

        assert!(
            arr.shape().iter().product::<usize>() > 0,
            "Array must not be empty."
        );

        LapackWrapper {
            arr,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return a triple (m, n, lda) for the Lapack interface.
    pub fn lapack_dims(&self) -> (i32, i32, i32) {
        let shape = self.arr.shape();
        let stride = self.arr.stride();

        let m = shape[0] as i32;
        let n = shape[1] as i32;
        assert_eq!(
            stride[0], 1,
            "Incorrect stride for Lapack. Stride[0] is {} but expected 1.",
            stride[0]
        );

        let lda = self.arr.stride()[1] as i32;

        (m, n, lda)
    }

    /// Return a slice to the underlying data.
    pub fn data(&self) -> &[Item]
    where
        ArrayImpl: RawAccess<Item = Item>,
    {
        self.arr.data()
    }

    /// Return a mutable slice to the underlying data.
    pub fn data_mut(&mut self) -> &mut [Item]
    where
        ArrayImpl: RawAccessMut<Item = Item>,
    {
        self.arr.data_mut()
    }
}

/// Interface to Lapack.
pub trait LapackOperations
where
    LapackWrapper<<Self::ArrayImpl as BaseItem>::Item, Self::ArrayImpl>: LapackInverse
        + LapackLu<Item = <Self::ArrayImpl as BaseItem>::Item, ArrayImpl = Self::ArrayImpl>,
    LuDecomposition<<Self::ArrayImpl as BaseItem>::Item, Self::ArrayImpl>:
        ComputedLu<Item = <Self::ArrayImpl as BaseItem>::Item, ArrayImpl = Self::ArrayImpl>,
{
    /// The array implementation type.
    type ArrayImpl: Shape<2> + Stride<2> + RawAccessMut;

    /// Interface to LAPACK operations.
    fn lapack(self) -> LapackWrapper<<Self::ArrayImpl as BaseItem>::Item, Self::ArrayImpl>;
}

impl<Item, ArrayImpl> LapackOperations for Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + Stride<2> + RawAccessMut<Item = Item>,
    LapackWrapper<Item, ArrayImpl>: LapackInverse + LapackLu<Item = Item, ArrayImpl = ArrayImpl>,
    LuDecomposition<Item, ArrayImpl>: ComputedLu<Item = Item, ArrayImpl = ArrayImpl>,
{
    fn lapack(self) -> LapackWrapper<<Self::ArrayImpl as BaseItem>::Item, Self::ArrayImpl> {
        LapackWrapper::new(self)
    }

    type ArrayImpl = ArrayImpl;
}
