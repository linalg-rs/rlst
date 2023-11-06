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

    pub fn get_l_resize<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + ResizeInPlace<2>,
    >(
        &self,
        mut arr: Array<Item, ArrayImplMut, 2>,
    ) {
        let m = self.arr.shape()[0];
        let n = self.arr.shape()[1];
        let k = std::cmp::min(m, n);

        arr.resize_in_place([m, k]);
        self.get_l(arr);
    }

    pub fn get_l<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        mut arr: Array<Item, ArrayImplMut, 2>,
    ) {
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
    }

    pub fn get_r_resize<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + ResizeInPlace<2>,
    >(
        &self,
        mut arr: Array<Item, ArrayImplMut, 2>,
    ) {
        let m = self.arr.shape()[0];
        let n = self.arr.shape()[1];
        let k = std::cmp::min(m, n);

        arr.resize_in_place([k, n]);
        self.get_r(arr);
    }

    pub fn get_r<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        mut arr: Array<Item, ArrayImplMut, 2>,
    ) {
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
            for row in 0..=std::cmp::min(col, k - 1) {
                arr[[row, col]] = self.arr.get_value([row, col]).unwrap();
            }
        }
    }

    pub fn get_p_resize<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + ResizeInPlace<2>,
    >(
        &self,
        mut arr: Array<Item, ArrayImplMut, 2>,
    ) {
        let m = self.arr.shape()[0];

        arr.resize_in_place([m, m]);
        self.get_p(arr);
    }

    fn get_perm(&self) -> Vec<usize> {
        let m = self.arr.shape()[0];
        // let n = self.arr.shape()[1];
        // let k = std::cmp::min(m, n);
        let ipiv: Vec<usize> = self.ipiv.iter().map(|&elem| (elem as usize) - 1).collect();

        let mut perm = (0..m).collect::<Vec<_>>();

        for (index, &elem) in ipiv.iter().enumerate() {
            perm.swap(index, elem);
        }

        perm
    }

    pub fn get_p<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        mut arr: Array<Item, ArrayImplMut, 2>,
    ) {
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

        let perm = self.get_perm();

        arr.set_zero();
        for col in 0..m {
            arr[[perm[col], col]] = <Item as One>::one();
        }
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

    use rlst_common::{assert_array_relative_eq, tools::PrettyPrint};

    use crate::rlst_dynamic_array2;

    use super::*;
    use crate::array::empty_array;

    #[test]
    fn test_lu_thick() {
        let dim = [8, 20];
        let mut arr = rlst_dynamic_array2!(f64, dim);

        arr.fill_from_seed_normally_distributed(0);
        let mut arr2 = rlst_dynamic_array2!(f64, dim);
        arr2.fill_from(arr.view());

        let lu = arr2.into_lu().unwrap();

        for (index, elem) in lu.get_perm().iter().enumerate() {
            println!("Ipiv[{}] = {}", index, elem)
        }

        let mut l_mat = empty_array::<f64, 2>();
        let mut r_mat = empty_array::<f64, 2>();
        let mut p_mat = empty_array::<f64, 2>();

        lu.get_l_resize(l_mat.view_mut());
        lu.get_r_resize(r_mat.view_mut());
        lu.get_p_resize(p_mat.view_mut());

        let res = crate::array::empty_array::<f64, 2>();

        let res = res.mult_into_resize(
            1.0,
            empty_array().mult_into_resize(1.0, p_mat, l_mat, 0.0),
            r_mat,
            0.0,
        );

        assert_array_relative_eq!(res, arr, 1E-12)
    }

    #[test]
    fn test_lu_square() {
        let dim = [12, 12];
        // let l_dim = [3, 3];
        // let r_dim = [3, 5];
        // let p_dim = [3, 3];
        let mut arr = rlst_dynamic_array2!(f64, dim);

        arr.fill_from_seed_normally_distributed(0);
        let mut arr2 = rlst_dynamic_array2!(f64, dim);
        arr2.fill_from(arr.view());

        let lu = arr2.into_lu().unwrap();

        for (index, elem) in lu.get_perm().iter().enumerate() {
            println!("Ipiv[{}] = {}", index, elem)
        }

        let mut l_mat = empty_array::<f64, 2>();
        let mut r_mat = empty_array::<f64, 2>();
        let mut p_mat = empty_array::<f64, 2>();

        lu.get_l_resize(l_mat.view_mut());
        lu.get_r_resize(r_mat.view_mut());
        lu.get_p_resize(p_mat.view_mut());

        l_mat.pretty_print();
        r_mat.pretty_print();
        p_mat.pretty_print();

        let res = crate::array::empty_array::<f64, 2>();

        let res = res.mult_into_resize(
            1.0,
            empty_array().mult_into_resize(1.0, p_mat, l_mat, 0.0),
            r_mat,
            0.0,
        );

        assert_array_relative_eq!(res, arr, 1E-12)
    }

    #[test]
    fn test_lu_thin() {
        let dim = [12, 8];
        // let l_dim = [3, 3];
        // let r_dim = [3, 5];
        // let p_dim = [3, 3];
        let mut arr = rlst_dynamic_array2!(f64, dim);

        arr.fill_from_seed_normally_distributed(0);
        let mut arr2 = rlst_dynamic_array2!(f64, dim);
        arr2.fill_from(arr.view());

        let lu = arr2.into_lu().unwrap();

        for (index, elem) in lu.get_perm().iter().enumerate() {
            println!("Ipiv[{}] = {}", index, elem)
        }

        let mut l_mat = empty_array::<f64, 2>();
        let mut r_mat = empty_array::<f64, 2>();
        let mut p_mat = empty_array::<f64, 2>();

        lu.get_l_resize(l_mat.view_mut());
        lu.get_r_resize(r_mat.view_mut());
        lu.get_p_resize(p_mat.view_mut());

        l_mat.pretty_print();
        r_mat.pretty_print();
        p_mat.pretty_print();

        let res = crate::array::empty_array::<f64, 2>();

        let res = res.mult_into_resize(
            1.0,
            empty_array().mult_into_resize(1.0, p_mat, l_mat, 0.0),
            r_mat,
            0.0,
        );

        assert_array_relative_eq!(res, arr, 1E-12)
    }
}
