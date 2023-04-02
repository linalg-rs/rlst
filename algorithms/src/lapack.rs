//! Interface to Lapack routines
pub mod lu_decomp;
pub use lapacke::Layout;
pub use rlst_common::types::{IndexType, RlstError, RlstResult};
use rlst_dense::types::Scalar;
use rlst_dense::{
    DataContainerMut, GenericBaseMatrixMut, LayoutType, MatrixTraitMut, SizeIdentifier,
};
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum TransposeMode {
    NoTrans = b'N',
    Trans = b'T',
    ConjugateTrans = b'C',
}

pub struct LapackData<
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    Mat: MatrixTraitMut<Item, RS, CS> + Sized,
> {
    pub mat: Mat,
    pub lda: i32,
    phantom_item: PhantomData<Item>,
    phantom_rs: PhantomData<RS>,
    phantom_cs: PhantomData<CS>,
}

pub fn check_lapack_stride(dim: (IndexType, IndexType), stride: (IndexType, IndexType)) -> bool {
    stride.0 == 1 && stride.1 >= std::cmp::max(1, dim.0)
}

pub trait AsLapack<
    Item: Scalar,
    Data: DataContainerMut<Item = Item>,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
>: MatrixTraitMut<Item, RS, CS> + Sized
{
    fn lapack(self) -> RlstResult<LapackData<Item, RS, CS, Self>> {
        let dim = self.layout().dim();
        if check_lapack_stride(self.layout().dim(), self.layout().stride()) {
            Ok(LapackData {
                mat: self,
                lda: dim.0 as i32,
                phantom_item: PhantomData,
                phantom_rs: PhantomData,
                phantom_cs: PhantomData,
            })
        } else {
            Err(RlstError::IncompatibleStride)
        }
    }
}

impl<Item: Scalar, Data: DataContainerMut<Item = Item>, RS: SizeIdentifier, CS: SizeIdentifier>
    AsLapack<Item, Data, RS, CS> for GenericBaseMatrixMut<Item, Data, RS, CS>
{
}
