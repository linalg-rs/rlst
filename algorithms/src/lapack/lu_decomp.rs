use crate::adapter::dense_matrix::{AsLapack, DenseContainer, DenseContainerInterfaceMut};
use crate::lapack::get_lapack_layout;
use crate::traits::lu_decomp::LUDecomp;
use lapacke;
use rlst_common::types::{RlstError, RlstResult};

pub struct LUDecompLapack<ContainerImpl: DenseContainerInterfaceMut> {
    data: AsLapack<ContainerImpl>,
    ipiv: Vec<i32>,
}

impl<ContainerImpl: DenseContainerInterfaceMut<T = f64>> AsLapack<ContainerImpl> {
    pub fn lu(self) -> RlstResult<LUDecompLapack<ContainerImpl>> {
        LUDecompLapack::new(self)
    }
}

impl<'a, ContainerImpl: DenseContainerInterfaceMut<T = f64>> LUDecompLapack<ContainerImpl> {
    pub fn new(mut data: AsLapack<ContainerImpl>) -> RlstResult<Self> {
        let dim = data.dim();
        let stride = data.stride();

        let m = dim.0 as i32;
        let n = dim.1 as i32;

        let mut ipiv: Vec<i32> = vec![0; std::cmp::min(dim.0, dim.1)];
        if let Ok((layout, lda)) = get_lapack_layout(stride) {
            let lda = lda as i32;
            let info = unsafe { lapacke::dgetrf(layout, m, n, data.data_mut(), lda, &mut ipiv) };
            if info == 0 {
                return Ok(Self { data, ipiv });
            } else {
                return Err(RlstError::LapackError(info));
            }
        } else {
            Err(RlstError::IncompatibleStride)
        }
    }
}

impl<ContainerImpl: DenseContainerInterfaceMut<T = f64>> LUDecomp
    for LUDecompLapack<ContainerImpl>
{
    type T = ContainerImpl::T;
    type ContainerImpl = ContainerImpl;

    fn data(&self) -> &[Self::T] {
        self.data.data()
    }

    fn dim(&self) -> (rlst_common::types::IndexType, rlst_common::types::IndexType) {
        self.data.dim()
    }

    fn solve(&self, rhs: &mut DenseContainer<Self::ContainerImpl>) {
        std::unimplemented!();
    }
}

#[cfg(test)]
use super::*;
use crate::adapter::adapter_traits::*;
use crate::adapter::rlst_dense_adapter::*;
use rlst_dense::{self, rlst_rand_mat};

#[test]
fn test_lu_decomp_f64() {
    let mut rlst_mat = rlst_rand_mat![f64, (5, 5)];

    let lu_decomp = rlst_mat.algorithms_mut().lapack().lu();

    let mut as_algorithm = rlst_mat.algorithms_mut();

    let mut as_lapack = as_algorithm.lapack();

    let lu_decomp = LUDecompLapack::new(as_lapack);
}
