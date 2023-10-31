//! LU Decomposition and linear system solves
use super::assert_lapack_stride;
use crate::array::Array;
use num::One;
use rlst_common::traits::*;
use rlst_common::types::*;
use rlst_lapack::Lapack;
use rlst_lapack::Trans;
use rlst_lapack::{Getrf, Getrs};

pub struct LuDecomposition<
    Item: Scalar + Lapack,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + Shape<2> + RawAccessMut<Item = Item>,
> {
    arr: Array<Item, ArrayImpl, 2>,
    ipiv: Vec<i32>,
}

impl<
        Item: Scalar + Lapack,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Item>,
    > LuDecomposition<Item, ArrayImpl>
{
    pub fn new(mut arr: Array<Item, ArrayImpl, 2>) -> RlstResult<Self> {
        let shape = arr.shape();
        let stride = arr.stride();

        assert_lapack_stride(stride);

        let dim = std::cmp::min(shape[0], shape[1]);
        let mut ipiv = vec![0; dim];
        let info = <Item as Getrf>::getrf(
            shape[0] as i32,
            shape[1] as i32,
            arr.data_mut(),
            stride[1] as i32,
            ipiv.as_mut_slice(),
        );

        match info {
            0 => Ok(Self { arr, ipiv }),
            _ => Err(RlstError::LapackError(info)),
        }
    }

    pub fn ipiv(&self) -> &[i32] {
        self.ipiv.as_slice()
    }

    pub fn solve_into<
        ArrayImplMut: RawAccessMut<Item = Item>
            + UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>,
    >(
        &self,
        trans: Trans,
        mut rhs: Array<Item, ArrayImplMut, 2>,
    ) -> RlstResult<Array<Item, ArrayImplMut, 2>> {
        assert_eq!(self.arr.shape()[0], self.arr.shape()[1]);
        let n = self.arr.shape()[0];
        assert_eq!(rhs.shape()[0], n);

        let nrhs = rhs.shape()[1];

        let arr_stride = self.arr.stride();
        let rhs_stride = rhs.stride();

        let lda = self.arr.stride()[1];
        let ldb = rhs.stride()[1];

        assert_lapack_stride(arr_stride);
        assert_lapack_stride(rhs_stride);

        let info = <Item as Getrs>::getrs(
            trans,
            n as i32,
            nrhs as i32,
            self.arr.data(),
            lda as i32,
            self.ipiv.as_slice(),
            rhs.data_mut(),
            ldb as i32,
        );

        match info {
            0 => Ok(rhs),
            _ => Err(RlstError::LapackError(info)),
        }
    }

    pub fn get_l<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        mut arr: Array<Item, ArrayImplMut, 2>,
    ) -> Array<Item, ArrayImplMut, 2> {
        let m = self.arr.shape()[0];
        let n = self.arr.shape()[1];
        let k = std::cmp::min(m, n);
        assert_eq!(
            arr.shape(),
            [m, k],
            "Require matrix with shape {} x {}. Given shape is {} x {}",
            m,
            k,
            arr.shape()[0],
            arr.shape()[1]
        );

        arr.set_zero();
        for col in 0..k {
            for row in col..m {
                if col == row {
                    arr[[row, col]] = <Item as One>::one();
                } else {
                    arr[[row, col]] = self.arr.get_value([row, col]).unwrap();
                }
            }
        }
        arr
    }

    pub fn get_r<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        mut arr: Array<Item, ArrayImplMut, 2>,
    ) -> Array<Item, ArrayImplMut, 2> {
        let m = self.arr.shape()[0];
        let n = self.arr.shape()[1];
        let k = std::cmp::min(m, n);
        assert_eq!(
            arr.shape(),
            [k, n],
            "Require matrix with shape {} x {}. Given shape is {} x {}",
            k,
            n,
            arr.shape()[0],
            arr.shape()[1]
        );

        arr.set_zero();
        for col in 0..n {
            for row in 0..=col {
                arr[[row, col]] = self.arr.get_value([row, col]).unwrap();
            }
        }
        arr
    }

    pub fn get_p<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        mut arr: Array<Item, ArrayImplMut, 2>,
    ) -> Array<Item, ArrayImplMut, 2> {
        let m = self.arr.shape()[0];
        assert_eq!(
            arr.shape(),
            [m, m],
            "Require matrix with shape {} x {}. Given shape is {} x {}",
            m,
            m,
            arr.shape()[0],
            arr.shape()[1]
        );

        arr.set_zero();
        for col in 0..m {
            arr[[self.ipiv[col] as usize, col]] = <Item as One>::one();
        }
        arr
    }
}

impl<
        Item: Scalar + Lapack,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>,
    > Array<Item, ArrayImpl, 2>
{
    pub fn into_lu(self) -> RlstResult<LuDecomposition<Item, ArrayImpl>> {
        LuDecomposition::new(self)
    }
}

#[cfg(test)]
mod test {

    use crate::rlst_dynamic_array2;

    use super::*;

    #[test]
    fn test_lu_thick() {
        let dim = [3, 5];
        let l_dim = [3, 3];
        let r_dim = [3, 5];
        let p_dim = [3, 3];
        let mut arr = rlst_dynamic_array2!(f64, dim);

        arr.fill_from_seed_normally_distributed(0);
        let mut arr2 = rlst_dynamic_array2!(f64, dim);
        arr2.fill_from(arr);

        let lu = arr2.into_lu().unwrap();

        let mut l_mat = rlst_dynamic_array2!(f64, l_dim);
        let mut r_mat = rlst_dynamic_array2!(f64, r_dim);
        let mut p_mat = rlst_dynamic_array2!(f64, p_dim);

        lu.get_l(l_mat.view_mut());
        lu.get_r(r_mat.view_mut());
        lu.get_p(p_mat.view_mut());
    }
}
