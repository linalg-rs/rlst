//! Interface to Lapack routines
pub mod lu_decomp;
pub use crate::adapter::dense_matrix::OrderType;
pub use rlst_common::types::{IndexType, RlstError, RlstResult};

// Given a tuple (row_stride, column_stride) return the layout and the Lapack LDA index.
pub fn get_lapack_layout(stride: (IndexType, IndexType)) -> RlstResult<(OrderType, IndexType)> {
    if stride.0 == 1 {
        // Column Major
        Ok((OrderType::ColumnMajor, stride.1))
    } else if stride.1 == 1 {
        // Row Major
        Ok((OrderType::RowMajor, stride.0))
    } else {
        Err(RlstError::IncompatibleStride)
    }
}
