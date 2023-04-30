//! Interface to Lapack routines.
pub mod lu_decomp;
pub use lapacke::Layout;
pub use rlst_common::types::{RlstError, RlstResult};
use rlst_dense::types::Scalar;
use rlst_dense::{
    DataContainerMut, GenericBaseMatrix, LayoutType, MatrixImplTraitMut, SizeIdentifier,
};
use std::marker::PhantomData;

/// Transposition mode for Lapack.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum TransposeMode {
    /// No transpose
    NoTrans = b'N',
    /// Transpose
    Trans = b'T',
    /// Conjugate Transpose
    ConjugateTrans = b'C',
}

/// A simple container to take ownership of a matrix for Lapack operations.
pub struct LapackData<
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    Mat: MatrixImplTraitMut<Item, RS, CS> + Sized,
> {
    /// The matrix on which to perform a Lapack operation.
    pub mat: Mat,
    /// The Lapack LDA parameter, which is the distance from one column to the next in memory.
    pub lda: i32,
    phantom_item: PhantomData<Item>,
    phantom_rs: PhantomData<RS>,
    phantom_cs: PhantomData<CS>,
}

/// Return true if a given stride is Lapack compatible. Otherwise, return false.
pub fn check_lapack_stride(dim: (usize, usize), stride: (usize, usize)) -> bool {
    stride.0 == 1 && stride.1 >= std::cmp::max(1, dim.0)
}

/// A trait that attaches to RLST Matrices and makes sure that data is represented
/// in a Lapack compatible format.
pub trait AsLapack<
    Item: Scalar,
    Data: DataContainerMut<Item = Item>,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
>: MatrixImplTraitMut<Item, RS, CS> + Sized
{
    /// Take ownership of a matrix and check that its layout is compatible with Lapack.
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
    AsLapack<Item, Data, RS, CS> for GenericBaseMatrix<Item, Data, RS, CS>
{
}
