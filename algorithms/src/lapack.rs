//! Interface to Lapack routines
pub mod lu_decomp;
pub use crate::adapter::dense_matrix::OrderType;
pub use lapacke::Layout;
pub use rlst_common::types::{IndexType, RlstError, RlstResult};

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum TransposeMode {
    NoTrans = b'N',
    Trans = b'T',
    ConjugateTrans = b'C',
}

// Given a tuple (row_stride, column_stride) return the layout and the Lapack LDA index.
pub fn get_lapack_layout(stride: (IndexType, IndexType)) -> RlstResult<(Layout, i32)> {
    if stride.0 == 1 {
        // Column Major
        Ok((Layout::ColumnMajor, stride.1 as i32))
    } else if stride.1 == 1 {
        // Row Major
        Ok((Layout::RowMajor, stride.0 as i32))
    } else {
        Err(RlstError::IncompatibleStride)
    }
}
