//! Dense Matrix Interface

pub use rlst_common::types::{IndexType, Scalar};
use rlst_dense::{
    base_matrix::BaseMatrix, ArbitraryStrideRowMajor, Dynamic, GenericBaseMatrixMut,
    VectorContainer,
};

pub enum OrderType {
    RowMajor,
    ColumnMajor,
}

/// A generic interface trait for dense matrices and vectors.
pub trait DenseContainerInterface {
    type T: Scalar;

    /// Return the dimension of the matrix/vector.
    fn dim(&self) -> (IndexType, IndexType);

    /// Return the number of elements.
    fn number_of_elements(&self) -> IndexType;

    /// Return the row and column stride.
    fn stride(&self) -> (IndexType, IndexType);

    /// Get a reference to an element without bounds check.
    unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> &Self::T;

    /// Get element using 1d numbering without bounds check.
    unsafe fn get1d_unchecked(&self, elem: IndexType) -> &Self::T;

    /// Get a reference to an element with bounds check.
    fn get(&self, row: IndexType, col: IndexType) -> Option<&Self::T>;

    /// Get a reference to an element using 1d numbering with bounds check.
    fn get1d(&self, elem: IndexType) -> Option<&Self::T>;

    /// Return a direct slice of the underlying data.
    fn data(&self) -> &[Self::T];
}

/// A mutable dense container interface.
pub trait DenseContainerInterfaceMut: DenseContainerInterface {
    /// Get a mutable reference to an element without bounds check.
    unsafe fn get_unchecked_mut(&mut self, row: IndexType, col: IndexType) -> &mut Self::T;

    /// Get mutable unchecked reference using 1d indexing.
    unsafe fn get1d_unchecked(&mut self, elem: IndexType) -> &mut Self::T;

    /// Get a mutable reference to an element with bounds check.
    fn get_mut(&mut self, row: IndexType, col: IndexType) -> Option<&mut Self::T>;

    /// Get a mutable reference using 1d indexing.
    fn get1d_mut(&mut self, elem: IndexType) -> Option<&mut Self::T>;

    /// Return a direct mutable slice of the underlying data.
    fn data_mut(&mut self) -> &mut [Self::T];
}

/// A simple dense container interface type.
pub struct DenseContainer<ContainerImpl: DenseContainerInterface> {
    data: ContainerImpl,
}

/// Create a new matrix.
pub fn new_container<T: Scalar>(
    rows: IndexType,
    cols: IndexType,
    order: OrderType,
) -> GenericBaseMatrixMut<T, ArbitraryStrideRowMajor, VectorContainer<T>, Dynamic, Dynamic> {
    let stride = match order {
        OrderType::ColumnMajor => (1, rows),
        OrderType::RowMajor => (cols, 1),
    };

    GenericBaseMatrixMut::new(BaseMatrix::new(
        VectorContainer::new(rows * cols),
        ArbitraryStrideRowMajor::new((rows, cols), stride),
    ))
}

/// Blas/Lapack routines will be attached to this type.
pub struct AsLapack<MatImpl: DenseContainerInterface> {
    data: MatImpl,
}

impl<MatImpl: DenseContainerInterface> DenseContainer<MatImpl> {
    pub fn lapack(self) -> AsLapack<MatImpl> {
        AsLapack { data: self.data }
    }

    /// Convert a dense container to an RLST matrix type.
    pub fn to_rlst<'b>(
        &'b self,
    ) -> rlst_dense::matrix::SliceMatrix<
        'b,
        MatImpl::T,
        rlst_dense::ArbitraryStrideRowMajor,
        rlst_dense::Dynamic,
        rlst_dense::Dynamic,
    > {
        let layout = rlst_dense::ArbitraryStrideRowMajor::new(self.dim(), self.stride());
        let data = rlst_dense::SliceContainer::<'b, MatImpl::T>::new(self.data());
        rlst_dense::SliceMatrix::from_data(data, layout)
    }
}

// impl<MatImpl: DenseContainerInterfaceMut> DenseContainer<MatImpl> {
//     pub fn lapack_mut<'a>(&'a mut self) -> AsLapackMut<'a, MatImpl> {
//         AsLapackMut {
//             mat: &mut self.data,
//         }
//     }

//     /// Convert a dense matrix to an RLST matrix type.
//     pub fn to_rlst_mut<'b>(
//         &'b mut self,
//     ) -> rlst_dense::matrix::SliceMatrixMut<
//         'b,
//         MatImpl::T,
//         rlst_dense::ArbitraryStrideRowMajor,
//         rlst_dense::Dynamic,
//         rlst_dense::Dynamic,
//     > {
//         let layout = rlst_dense::ArbitraryStrideRowMajor::new(self.dim(), self.stride());
//         let data = rlst_dense::SliceContainerMut::<'b, MatImpl::T>::new(self.data_mut());
//         rlst_dense::SliceMatrixMut::from_data(data, layout)
//     }
// }

// impl<'a, MatImpl: DenseMatrixInterfaceMut> DenseMatrixMut<'a, MatImpl> {
//     pub fn lapack_mut<'b>(&'b mut self) -> AsLapackMut<'b, MatImpl> {
//         AsLapackMut { mat: self.mat }
//     }

//     /// Convert a mutable dense matrix to a mutable RLST matrix.
//     pub fn to_rlst_mut<'b>(
//         &'b mut self,
//     ) -> rlst_dense::matrix::SliceMatrixMut<
//         'b,
//         MatImpl::T,
//         rlst_dense::ArbitraryStrideRowMajor,
//         rlst_dense::Dynamic,
//         rlst_dense::Dynamic,
//     > {
//         let layout = rlst_dense::ArbitraryStrideRowMajor::new(self.dim(), self.stride());
//         let data = rlst_dense::SliceContainerMut::<'b, MatImpl::T>::new(self.data_mut());
//         rlst_dense::SliceMatrixMut::from_data(data, layout)
//     }
// }

macro_rules! implement_dense_container {
    ($name:ident) => {
        impl<ContainerImpl: DenseContainerInterface> $name<ContainerImpl> {
            pub fn new(data: ContainerImpl) -> Self {
                Self { data }
            }

            #[inline]
            pub fn dim(&self) -> (IndexType, IndexType) {
                self.data.dim()
            }

            #[inline]
            pub fn stride(&self) -> (IndexType, IndexType) {
                self.data.stride()
            }

            #[inline]
            pub unsafe fn get_unchecked(
                &self,
                row: IndexType,
                col: IndexType,
            ) -> &ContainerImpl::T {
                self.data.get_unchecked(row, col)
            }

            #[inline]
            pub fn get(&self, row: IndexType, col: IndexType) -> Option<&ContainerImpl::T> {
                self.data.get(row, col)
            }

            #[inline]
            pub fn data(&self) -> &[ContainerImpl::T] {
                self.data.data()
            }
        }
    };
}

macro_rules! implement_dense_container_mut {
    ($name:ident) => {
        impl<ContainerImpl: DenseContainerInterfaceMut> $name<ContainerImpl> {
            #[inline]
            pub unsafe fn get_unchecked_mut(
                &mut self,
                row: IndexType,
                col: IndexType,
            ) -> &mut ContainerImpl::T {
                self.data.get_unchecked_mut(row, col)
            }

            #[inline]
            pub fn get_mut(
                &mut self,
                row: IndexType,
                col: IndexType,
            ) -> Option<&mut ContainerImpl::T> {
                self.data.get_mut(row, col)
            }

            #[inline]
            pub fn data_mut(&mut self) -> &mut [ContainerImpl::T] {
                self.data.data_mut()
            }
        }
    };
}

implement_dense_container!(DenseContainer);
implement_dense_container!(AsLapack);

implement_dense_container_mut!(DenseContainer);
implement_dense_container_mut!(AsLapack);

// impl<ContainerImpl: DenseContainerInterface> DenseContainer<ContainerImpl> {
//     pub fn new(data: ContainerImpl) -> Self {
//         Self { data }
//     }

//     #[inline]
//     pub fn dim(&self) -> (IndexType, IndexType) {
//         self.data.dim()
//     }

//     #[inline]
//     pub fn stride(&self) -> (IndexType, IndexType) {
//         self.data.stride()
//     }

//     #[inline]
//     pub unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> &ContainerImpl::T {
//         self.data.get_unchecked(row, col)
//     }

//     #[inline]
//     pub fn get(&self, row: IndexType, col: IndexType) -> Option<&ContainerImpl::T> {
//         self.data.get(row, col)
//     }

//     #[inline]
//     pub fn data(&self) -> &[ContainerImpl::T] {
//         self.data.data()
//     }
// }

// impl<'a, ContainerImpl: DenseContainerInterfaceMut> DenseContainer<ContainerImpl> {
//     #[inline]
//     pub unsafe fn get_unchecked_mut(
//         &mut self,
//         row: IndexType,
//         col: IndexType,
//     ) -> &mut ContainerImpl::T {
//         self.data.get_unchecked_mut(row, col)
//     }

//     #[inline]
//     pub fn get_mut(&mut self, row: IndexType, col: IndexType) -> Option<&mut ContainerImpl::T> {
//         self.data.get_mut(row, col)
//     }

//     #[inline]
//     pub fn data_mut(&mut self) -> &mut [ContainerImpl::T] {
//         self.data.data_mut()
//     }
// }

// macro_rules! implement_dense_container_with_lifetime {
//     ($mat:ident) => {
//         impl<'a, MatImpl: DenseContainerInterface> $mat<'a, MatImpl> {
//             #[inline]
//             pub fn dim(&self) -> (IndexType, IndexType) {
//                 self.mat.dim()
//             }

//             #[inline]
//             pub fn stride(&self) -> (IndexType, IndexType) {
//                 self.mat.stride()
//             }

//             #[inline]
//             pub unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> &MatImpl::T {
//                 self.mat.get_unchecked(row, col)
//             }

//             #[inline]
//             pub fn get(&self, row: IndexType, col: IndexType) -> Option<&MatImpl::T> {
//                 self.mat.get(row, col)
//             }

//             #[inline]
//             pub fn data(&self) -> &[MatImpl::T] {
//                 self.mat.data()
//             }
//         }
//     };
// }

// macro_rules! implement_dense_container_with_lifetime_mut {
//     ($mat:ident) => {
//         impl<'a, MatImpl: DenseContainerInterfaceMut> $mat<'a, MatImpl> {
//             pub fn dim(&self) -> (IndexType, IndexType) {
//                 self.mat.dim()
//             }
//             pub fn stride(&self) -> (IndexType, IndexType) {
//                 self.mat.stride()
//             }

//             pub unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> &MatImpl::T {
//                 self.mat.get_unchecked(row, col)
//             }
//             pub fn get(&self, row: IndexType, col: IndexType) -> Option<&MatImpl::T> {
//                 self.mat.get(row, col)
//             }

//             pub fn data(&self) -> &[MatImpl::T] {
//                 self.mat.data()
//             }

//             pub unsafe fn get_unchecked_mut(
//                 &mut self,
//                 row: IndexType,
//                 col: IndexType,
//             ) -> &mut MatImpl::T {
//                 self.mat.get_unchecked_mut(row, col)
//             }

//             pub fn get_mut(&mut self, row: IndexType, col: IndexType) -> Option<&mut MatImpl::T> {
//                 self.mat.get_mut(row, col)
//             }

//             pub fn data_mut(&mut self) -> &mut [MatImpl::T] {
//                 self.mat.data_mut()
//             }
//         }
//     };
// }

// implement_dense_container_with_lifetime!(AsLapack);
// implement_dense_container_with_lifetime_mut!(AsLapackMut);
