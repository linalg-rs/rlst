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

pub use crate::rlst_rank1_array;

pub use rlst_proc_macro::measure_duration;
pub use rlst_proc_macro::rlst_static_array;
pub use rlst_proc_macro::rlst_static_type;

pub use std::ops::Add;
pub use std::ops::Neg;
pub use std::ops::Sub;

pub use crate::dense::gemm::Gemm;

pub use crate::dense::tools::PrettyPrint;

pub use crate::dense::traits::{
    AsOperatorApply, ChunkedAccess, RandomAccessByRef, RandomAccessByValue, RandomAccessMut,
    RawAccess, RawAccessMut, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue,
    UnsafeRandomAccessMut,
};

pub use crate::dense::batched_gemm::{BatchedGemm, DefaultCpuBatchedGemm};

pub use crate::dense::linalg::LinAlg;

pub use crate::dense::traits::{
    AijIterator, AsMultiIndex, DefaultIterator, DefaultIteratorMut, MultInto, MultIntoResize,
    NumberOfElements, ResizeInPlace, Shape, Stride,
};

pub use crate::dense::types::{
    c32, c64, DataChunk, RlstBase, RlstError, RlstNum, RlstResult, RlstScalar, Side, TransMode,
    TriangularType,
};

pub use crate::dense::base_array::BaseArray;
pub use crate::dense::data_container::{
    ArrayContainer, SliceContainer, SliceContainerMut, VectorContainer,
};

pub use crate::dense::array::empty_axis::AxisPosition;

pub use crate::dense::linalg::interpolative_decomposition::{IdDecomposition, MatrixId};
pub use crate::dense::linalg::inverse::MatrixInverse;
pub use crate::dense::linalg::lu::{LuDecomposition, MatrixLuDecomposition};
pub use crate::dense::linalg::null_space::{MatrixNull, NullSpace};
pub use crate::dense::linalg::pseudo_inverse::MatrixPseudoInverse;
pub use crate::dense::linalg::qr::{MatrixQr, MatrixQrDecomposition, QrDecomposition};
pub use crate::dense::linalg::svd::{MatrixSvd, SvdMode};
pub use crate::dense::linalg::triangular_arrays::{
    Triangular, TriangularMatrix, TriangularOperations,
};

pub use crate::dense::array::rank1_array::Rank1Array;

pub use crate::dense::array::{DynamicArray, SliceArray, SliceArrayMut};

pub use crate::dense::simd::{RlstSimd, SimdFor};

#[cfg(feature = "mpi")]
pub use crate::sparse::{
    distributed_vector::DistributedVector, sparse_mat::distributed_csr_mat::DistributedCsrMatrix,
};
#[cfg(feature = "mpi")]
pub use bempp_distributed_tools::IndexLayout;

pub use crate::sparse::sparse_mat::csc_mat::CscMatrix;
pub use crate::sparse::sparse_mat::csr_mat::CsrMatrix;

pub use crate::operator::interface::{ArrayVectorSpace, ArrayVectorSpaceElement, MatrixOperator};
#[cfg(feature = "mpi")]
pub use crate::operator::interface::{
    DistributedArrayVectorSpace, DistributedArrayVectorSpaceElement,
};

pub use crate::operator::element::{
    Element, ElementContainer, ElementContainerMut, ScalarTimesElement,
};
pub use crate::operator::operations::conjugate_gradients::CgIteration;
pub use crate::operator::operations::gmres::GmresIteration;
pub use crate::operator::operations::modified_gram_schmidt::ModifiedGramSchmidt;
pub use crate::operator::space::frame::{Frame, VectorFrame};
pub use crate::operator::OperatorLeftScalarMul;
pub use crate::operator::{
    zero_element, DualSpace, IndexableSpace, InnerProductSpace, LinearSpace, NormedSpace,
};
pub use crate::operator::{AsApply, Operator, OperatorBase};
pub use crate::operator::{ElementImpl, ElementView, ElementViewMut};

pub use crate::operator::abstract_operator::ScalarTimesOperator;

pub use crate::tracing::Tracing;
