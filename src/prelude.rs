//! Import everything from here to get the standard RLST functionality

pub use crate::dense::array::empty_array;
pub use crate::dense::array::Array;

pub use crate::rlst_dynamic_array1;
pub use crate::rlst_dynamic_array2;
pub use crate::rlst_dynamic_array3;
pub use crate::rlst_dynamic_array4;
pub use crate::rlst_dynamic_array5;

pub use crate::rlst_array_from_slice1;
pub use crate::rlst_array_from_slice2;
pub use crate::rlst_array_from_slice3;
pub use crate::rlst_array_from_slice4;
pub use crate::rlst_array_from_slice5;

pub use crate::rlst_array_from_slice_mut1;
pub use crate::rlst_array_from_slice_mut2;
pub use crate::rlst_array_from_slice_mut3;
pub use crate::rlst_array_from_slice_mut4;
pub use crate::rlst_array_from_slice_mut5;

pub use rlst_proc_macro::rlst_static_array;
pub use rlst_proc_macro::rlst_static_type;

pub use crate::dense::gemm::Gemm;

pub use crate::dense::traits::{
    ChunkedAccess, RandomAccessByRef, RandomAccessByValue, RandomAccessMut, RawAccess,
    RawAccessMut, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

pub use crate::dense::traits::{
    AijIterator, AsMultiIndex, DefaultIterator, DefaultIteratorMut, MultInto, MultIntoResize,
    NumberOfElements, ResizeInPlace, Shape, Stride,
};

pub use crate::dense::types::{c32, c64, DataChunk, RlstError, RlstResult, RlstScalar, TransMode};

pub use crate::dense::base_array::BaseArray;
pub use crate::dense::data_container::{
    ArrayContainer, SliceContainer, SliceContainerMut, VectorContainer,
};

pub use crate::dense::array::empty_axis::AxisPosition;

pub use crate::dense::linalg::inverse::MatrixInverse;
pub use crate::dense::linalg::lu::{LuDecomposition, LuTrans, MatrixLuDecomposition};
pub use crate::dense::linalg::pseudo_inverse::MatrixPseudoInverse;
pub use crate::dense::linalg::qr::{MatrixQrDecomposition, QrDecomposition};
pub use crate::dense::linalg::svd::{MatrixSvd, SvdMode};

pub use crate::dense::array::{DynamicArray, SliceArray, SliceArrayMut};

#[cfg(feature = "mpi")]
pub use crate::sparse::index_layout::DefaultMpiIndexLayout;

#[cfg(feature = "mpi")]
pub use crate::sparse::sparse_mat::mpi_csr_mat::MpiCsrMatrix;

pub use crate::sparse::index_layout::DefaultSerialIndexLayout;
pub use crate::sparse::sparse_mat::csc_mat::CscMatrix;
pub use crate::sparse::sparse_mat::csr_mat::CsrMatrix;
pub use crate::sparse::traits::index_layout::IndexLayout;

pub use crate::operator::interface::{
    ArrayVectorSpace, ArrayVectorSpaceElement, CscMatrixOperator, CsrMatrixOperator,
    DenseMatrixOperator,
};

pub use crate::operator::operations::conjugate_gradients::CgIteration;
pub use crate::operator::operations::modified_gram_schmidt::ModifiedGramSchmidt;
pub use crate::operator::space::frame::{Frame, VectorFrame};
pub use crate::operator::{AsApply, OperatorBase};
pub use crate::operator::{DualSpace, IndexableSpace, InnerProductSpace, LinearSpace, NormedSpace};
pub use crate::operator::{Element, ElementView, ElementViewMut};
