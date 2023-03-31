//! Interface to Lapack routines
pub mod lu_decomp;
pub use lapacke::Layout;
pub use rlst_common::types::{IndexType, RlstError, RlstResult};
use rlst_dense::types::Scalar;
use rlst_dense::Layout as _;
use rlst_dense::{
    DataContainerMut, DefaultLayout, GenericBaseMatrixMut, LayoutType, MatrixD, MatrixTrait,
    SizeIdentifier, UnsafeRandomAccessMut, VectorContainer,
};

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum TransposeMode {
    NoTrans = b'N',
    Trans = b'T',
    ConjugateTrans = b'C',
}

pub struct LapackData<Item: Scalar, RS: SizeIdentifier, CS: SizeIdentifier> {
    mat: GenericBaseMatrixMut<Item, VectorContainer<Item>, RS, CS>,
    lda: i32,
}

pub trait AsLapack<Item: Scalar, RS: SizeIdentifier, CS: SizeIdentifier>:
    MatrixTrait<Item, RS, CS> + rlst_dense::Layout<Impl = DefaultLayout>
{
    fn lapack(self) -> LapackData<Item, VectorContainer<Item>, RS, CS> {
        let dim = self.layout().dim();
        let mut result = MatrixD::<Item>::zeros_from_dim(dim.0, dim.1);
        unsafe {
            for row in 0..dim.0 {
                for col in 0..dim.1 {
                    *result.get_unchecked_mut(row, col) = self.get_value_unchecked(row, col);
                }
            }
        }
        LapackData {
            mat: result,
            lda: dim.0 as i32,
        }
    }
}

impl<Item: Scalar, RS: SizeIdentifier, CS: SizeIdentifier> AsLapack<Item, RS, CS>
    for GenericBaseMatrixMut<Item, VectorContainer<Item>, RS, CS>
{
    fn lapack(self) -> LapackData<Item, VectorContainer<Item>, RS, CS> {
        let mat_t;

        let stride = self.layout().stride();
        let dim = self.layout().dim();

        if stride == (1, dim.0) {
            mat_t = self;
        } else {
            mat_t = self.eval();
        }

        LapackData {
            mat: mat_t,
            lda: dim.0 as i32,
        }
    }
}

// Given a tuple (row_stride, column_stride) return the layout and the Lapack LDA index.
// pub fn to_lapack_layout<(stride: (IndexType, IndexType)) -> RlstResult<(Layout, i32)> {
//     if stride.0 == 1 {
//         // Column Major
//         Ok((Layout::ColumnMajor, stride.1 as i32))
//     } else if stride.1 == 1 {
//         // Row Major
//         Ok((Layout::RowMajor, stride.0 as i32))
//     } else {
//         Err(RlstError::IncompatibleStride)
//     }
// }
