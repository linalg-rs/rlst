pub use crate::dense_matrix_traits::{DenseMatrixAccess, DenseMatrixAccessMut};
pub use rlst_common::types::{IndexType, Scalar};

pub struct DenseMatrix<'a, MatImpl: DenseMatrixAccess> {
    mat: &'a MatImpl,
}

pub struct DenseMatrixMut<'a, MatImpl: DenseMatrixAccessMut> {
    mat: &'a mut MatImpl,
}

pub struct AsLapack<'a, MatImpl: DenseMatrixAccess> {
    mat: &'a MatImpl,
}

pub struct AsLapackMut<'a, MatImpl: DenseMatrixAccessMut> {
    mat: &'a mut MatImpl,
}

impl<'a, MatImpl: DenseMatrixAccess> DenseMatrix<'a, MatImpl> {
    pub fn as_lapack<'b>(&'b self) -> AsLapack<'b, MatImpl> {
        AsLapack { mat: self.mat }
    }

    pub fn as_rlst<'b>(
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

impl<'a, MatImpl: DenseMatrixAccessMut> DenseMatrixMut<'a, MatImpl> {
    pub fn as_lapack_mut<'b>(&'b mut self) -> AsLapackMut<'b, MatImpl> {
        AsLapackMut { mat: self.mat }
    }

    pub fn as_rlst_mut<'b>(
        &'b mut self,
    ) -> rlst_dense::matrix::SliceMatrixMut<
        'b,
        MatImpl::T,
        rlst_dense::ArbitraryStrideRowMajor,
        rlst_dense::Dynamic,
        rlst_dense::Dynamic,
    > {
        let layout = rlst_dense::ArbitraryStrideRowMajor::new(self.dim(), self.stride());
        let data = rlst_dense::SliceContainerMut::<'b, MatImpl::T>::new(self.data_mut());
        rlst_dense::SliceMatrixMut::from_data(data, layout)
    }
}

macro_rules! implement_dense_matrix {
    ($mat:ident) => {
        impl<'a, MatImpl: DenseMatrixAccess> $mat<'a, MatImpl> {
            #[inline]
            pub fn dim(&self) -> (IndexType, IndexType) {
                self.mat.dim()
            }

            #[inline]
            pub fn stride(&self) -> (IndexType, IndexType) {
                self.mat.stride()
            }

            #[inline]
            pub unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> &MatImpl::T {
                self.mat.get_unchecked(row, col)
            }

            #[inline]
            pub fn get(&self, row: IndexType, col: IndexType) -> Option<&MatImpl::T> {
                self.mat.get(row, col)
            }

            #[inline]
            pub fn data(&self) -> &[MatImpl::T] {
                self.mat.data()
            }
        }
    };
}

macro_rules! implement_dense_matrix_mut {
    ($mat:ident) => {
        impl<'a, MatImpl: DenseMatrixAccessMut> $mat<'a, MatImpl> {
            pub fn dim(&self) -> (IndexType, IndexType) {
                self.mat.dim()
            }
            pub fn stride(&self) -> (IndexType, IndexType) {
                self.mat.stride()
            }

            pub unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> &MatImpl::T {
                self.mat.get_unchecked(row, col)
            }
            pub fn get(&self, row: IndexType, col: IndexType) -> Option<&MatImpl::T> {
                self.mat.get(row, col)
            }

            pub fn data(&self) -> &[MatImpl::T] {
                self.mat.data()
            }

            pub unsafe fn get_unchecked_mut(
                &mut self,
                row: IndexType,
                col: IndexType,
            ) -> &mut MatImpl::T {
                self.mat.get_unchecked_mut(row, col)
            }

            pub fn get_mut(&mut self, row: IndexType, col: IndexType) -> Option<&mut MatImpl::T> {
                self.mat.get_mut(row, col)
            }

            pub fn data_mut(&mut self) -> &mut [MatImpl::T] {
                self.mat.data_mut()
            }
        }
    };
}

implement_dense_matrix!(DenseMatrix);
implement_dense_matrix!(AsLapack);

implement_dense_matrix_mut!(DenseMatrixMut);
implement_dense_matrix_mut!(AsLapackMut);
