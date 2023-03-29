use crate::adapter::dense_matrix::{AsLapackMut, DenseContainer, DenseContainerInterfaceMut};
use crate::lapack::get_lapack_layout;
use crate::traits::lu_decomp::LUDecomp;
use rlst_common::types::{RlstError, RlstResult};

use lapack_src;
use lapack_sys;

pub struct LUDecompLapack<'a, ContainerImpl: DenseContainerInterfaceMut> {
    data: AsLapackMut<'a, ContainerImpl>,
    ipiv: Vec<i32>,
}

impl<'a, ContainerImpl: DenseContainerInterfaceMut<T = f64>> LUDecompLapack<'a, ContainerImpl> {
    pub fn new(mut data: AsLapackMut<'a, ContainerImpl>) -> RlstResult<Self> {
        let dim = data.dim();
        let stride = data.stride();

        let m = dim.0 as i32;
        let n = dim.1 as i32;

        let mut ipiv: Vec<i32> = vec![0; std::cmp::min(dim.0, dim.1)];
        if let Ok((layout, lda)) = get_lapack_layout(stride) {
            let mut info: i32 = 0;
            let lda = lda as i32;
            unsafe {
                lapack_sys::dgetrf_(
                    &m,
                    &n,
                    data.data_mut().as_mut_ptr(),
                    &lda,
                    ipiv.as_mut_slice().as_mut_ptr(),
                    &mut info,
                );
            };
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

impl<'a, ContainerImpl: DenseContainerInterfaceMut<T = f64>> LUDecomp
    for LUDecompLapack<'a, ContainerImpl>
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

    let mut as_algorithm = rlst_mat.algorithms_mut();

    let mut as_lapack = as_algorithm.lapack_mut();

    let lu_decomp = LUDecompLapack::new(as_lapack);
}
