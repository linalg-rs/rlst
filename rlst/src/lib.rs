//! Interface to the rlst library

extern crate blas_src;
extern crate lapack_src;

pub use rlst_dense as dense;

pub use rlst_dense::array::empty_array;
pub use rlst_dense::array::Array;

pub use rlst_dense::rlst_dynamic_array1;
pub use rlst_dense::rlst_dynamic_array2;
pub use rlst_dense::rlst_dynamic_array3;
pub use rlst_dense::rlst_dynamic_array4;

pub use rlst_dense::rlst_array_from_slice1;
pub use rlst_dense::rlst_array_from_slice2;
pub use rlst_dense::rlst_array_from_slice3;
pub use rlst_dense::rlst_array_from_slice4;

pub use rlst_dense::rlst_array_from_slice_mut1;
pub use rlst_dense::rlst_array_from_slice_mut2;
pub use rlst_dense::rlst_array_from_slice_mut3;
pub use rlst_dense::rlst_array_from_slice_mut4;

pub use rlst_proc_macro::rlst_static_array;
pub use rlst_proc_macro::rlst_static_type;

pub use rlst_dense::gemm::Gemm;
pub use rlst_dense::traits::*;
pub use rlst_dense::types::*;

pub use rlst_dense::base_array::BaseArray;
pub use rlst_dense::data_container::{
    ArrayContainer, SliceContainer, SliceContainerMut, VectorContainer,
};

pub use rlst_dense::array::empty_axis::AxisPosition;

pub use rlst_dense::linalg::inverse::MatrixInverse;
pub use rlst_dense::linalg::lu::{LuDecomposition, LuTrans, MatrixLuDecomposition};
pub use rlst_dense::linalg::pseudo_inverse::MatrixPseudoInverse;
pub use rlst_dense::linalg::qr::{MatrixQrDecomposition, QrDecomposition};
pub use rlst_dense::linalg::svd::{MatrixSvd, SvdMode};

pub use rlst_dense::array::{DynamicArray, SliceArray, SliceArrayMut};

pub use rlst_sparse::index_layout::DefaultMpiIndexLayout;
pub use rlst_sparse::index_layout::DefaultSerialIndexLayout;
pub use rlst_sparse::sparse::csc_mat::CscMatrix;
pub use rlst_sparse::sparse::csr_mat::CsrMatrix;
pub use rlst_sparse::sparse::mpi_csr_mat::MpiCsrMatrix;
pub use rlst_sparse::traits::index_layout::IndexLayout;

pub use rlst_sparse as sparse;

pub use rlst_operator::*;
