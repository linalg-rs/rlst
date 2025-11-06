//! Traits used by RLST.

pub mod abstract_operator;
pub mod accessors;
pub mod base_operations;
pub mod data_container;
#[cfg(feature = "mpi")]
pub mod distributed;
pub mod io;
pub mod iterators;
pub mod linalg;
pub mod linear_space;
pub mod number_relations;
pub mod number_traits;
pub mod rlst_num;
pub mod sparse;

pub use accessors::*;
pub use base_operations::*;
pub use data_container::*;
#[cfg(feature = "mpi")]
pub use distributed::*;
pub use io::*;
pub use iterators::*;
pub use linalg::*;
pub use linear_space::*;
pub use number_relations::*;
pub use number_traits::*;
pub use rlst_num::*;
pub use sparse::*;

/// Default trait for Arrays that can return values.
pub trait ValueArrayImpl<Item, const NDIM: usize>:
    UnsafeRandom1DAccessByValue<Item = Item>
    + UnsafeRandomAccessByValue<NDIM, Item = Item>
    + Shape<NDIM>
{
}

impl<A, Item, const NDIM: usize> ValueArrayImpl<Item, NDIM> for A where
    A: UnsafeRandom1DAccessByValue<Item = Item>
        + UnsafeRandomAccessByValue<NDIM, Item = Item>
        + Shape<NDIM>
{
}

/// Default trait for Arrays that can return references.
pub trait RefArrayImpl<Item, const NDIM: usize>:
    ValueArrayImpl<Item, NDIM>
    + UnsafeRandom1DAccessByRef<Item = Item>
    + UnsafeRandomAccessByRef<NDIM, Item = Item>
{
}

impl<A, Item, const NDIM: usize> RefArrayImpl<Item, NDIM> for A where
    A: ValueArrayImpl<Item, NDIM>
        + UnsafeRandom1DAccessByRef<Item = Item>
        + UnsafeRandomAccessByRef<NDIM, Item = Item>
{
}

/// Default trait for mutable arrays.
pub trait MutableArrayImpl<Item, const NDIM: usize>:
    RefArrayImpl<Item, NDIM>
    + UnsafeRandom1DAccessMut<Item = Item>
    + UnsafeRandomAccessMut<NDIM, Item = Item>
{
}

impl<A, Item, const NDIM: usize> MutableArrayImpl<Item, NDIM> for A where
    A: RefArrayImpl<Item, NDIM>
        + UnsafeRandom1DAccessMut<Item = Item>
        + UnsafeRandomAccessMut<NDIM, Item = Item>
{
}

/// Default trait for arrays that allow RawAccess.
pub trait RawAccessArrayImpl<Item, const NDIM: usize>:
    ValueArrayImpl<Item, NDIM> + RawAccess<Item = Item> + Stride<NDIM>
{
}

impl<A, Item, const NDIM: usize> RawAccessArrayImpl<Item, NDIM> for A where
    A: ValueArrayImpl<Item, NDIM> + RawAccess<Item = Item> + Stride<NDIM>
{
}

/// Default trait for arrays that allow mutable raw access.
pub trait MutableRawAccessArrayImpl<Item, const NDIM: usize>:
    RawAccessArrayImpl<Item, NDIM> + RawAccessMut<Item = Item>
{
}

impl<A, Item, const NDIM: usize> MutableRawAccessArrayImpl<Item, NDIM> for A where
    A: RawAccessArrayImpl<Item, NDIM> + RawAccessMut<Item = Item>
{
}
