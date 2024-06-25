//! A rank1 update of a two-dimensional array.

use crate::dense::layout::convert_1d_nd_from_shape;
use crate::dense::types::{c32, c64};

use crate::{Array, ChunkedAccess, DataChunk, RlstScalar, Shape, UnsafeRandomAccessByValue};

/// Rank-1 Update of a two-dimensional array.
pub struct Rank1UpdateSum<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>,
    ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
> {
    arr: Array<Item, ArrayImpl, 2>,
    u: Array<Item, ArrayImplU, 1>,
    v: Array<Item, ArrayImplV, 1>,
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    > Rank1UpdateSum<Item, ArrayImpl, ArrayImplU, ArrayImplV>
{
    /// Initialize a rank 1 update of the form `arr + uv^T`.
    pub fn new(
        arr: Array<Item, ArrayImpl, 2>,
        u: Array<Item, ArrayImplU, 1>,
        v: Array<Item, ArrayImplV, 1>,
    ) -> Self {
        assert_eq!(arr.shape()[0], u.shape()[0]);
        assert_eq!(arr.shape()[1], v.shape()[0]);
        Self { arr, u, v }
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    > UnsafeRandomAccessByValue<2> for Rank1UpdateSum<Item, ArrayImpl, ArrayImplU, ArrayImplV>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; 2]) -> Self::Item {
        if coe::is_same::<Item, f64>() {
            let u_val: f64 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: f64 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            let arr_val: f64 = coe::coerce_static(self.arr.get_value_unchecked(multi_index));
            coe::coerce_static(u_val.mul_add(v_val, arr_val))
        } else if coe::is_same::<Item, f32>() {
            let u_val: f32 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: f32 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            let arr_val: f32 = coe::coerce_static(self.arr.get_value_unchecked(multi_index));
            coe::coerce_static(u_val.mul_add(v_val, arr_val))
        } else if coe::is_same::<Item, c32>() {
            let u_val: c32 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: c32 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            let arr_val: c32 = coe::coerce_static(self.arr.get_value_unchecked(multi_index));
            coe::coerce_static(c32::new(
                arr_val.re() + u_val.re().mul_add(v_val.re(), -u_val.im() * v_val.im()),
                arr_val.im() + u_val.re().mul_add(v_val.im(), u_val.im() * v_val.re()),
            ))
        } else if coe::is_same::<Item, c64>() {
            let u_val: c64 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: c64 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            let arr_val: c64 = coe::coerce_static(self.arr.get_value_unchecked(multi_index));
            coe::coerce_static(c64::new(
                arr_val.re() + u_val.re().mul_add(v_val.re(), -u_val.im() * v_val.im()),
                arr_val.im() + u_val.re().mul_add(v_val.im(), u_val.im() * v_val.re()),
            ))
        } else {
            panic!("Unknown type");
        }
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    > Shape<2> for Rank1UpdateSum<Item, ArrayImpl, ArrayImplU, ArrayImplV>
{
    fn shape(&self) -> [usize; 2] {
        self.arr.shape()
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + ChunkedAccess<N, Item = Item>,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        const N: usize,
    > ChunkedAccess<N> for Rank1UpdateSum<Item, ArrayImpl, ArrayImplU, ArrayImplV>
{
    type Item = Item;

    fn get_chunk(&self, chunk_index: usize) -> Option<crate::DataChunk<Self::Item, N>> {
        if let Some(chunk) = self.arr.get_chunk(chunk_index) {
            let mut output_data = [<Item as num::Zero>::zero(); N];
            let mut u_data = [<Item as num::Zero>::zero(); N];
            let mut v_data = [<Item as num::Zero>::zero(); N];
            let shape = self.arr.shape();
            for (index, u_elem, v_elem) in
                itertools::izip!(0..chunk.valid_entries, u_data.iter_mut(), v_data.iter_mut(),)
            {
                let multi_index = convert_1d_nd_from_shape(index + chunk.start_index, shape);
                *u_elem = unsafe { self.u.get_value_unchecked([multi_index[0]]) };
                *v_elem = unsafe { self.v.get_value_unchecked([multi_index[1]]) };
            }

            if coe::is_same::<Item, f64>() {
                for (out, &data, &u_elem, &v_elem) in itertools::izip!(
                    output_data.as_mut_slice(),
                    chunk.data.as_slice(),
                    u_data.as_slice(),
                    v_data.as_slice()
                ) {
                    let data: f64 = coe::coerce_static(data);
                    let u_elem: f64 = coe::coerce_static(u_elem);
                    let v_elem: f64 = coe::coerce_static(v_elem);

                    *out = coe::coerce_static(u_elem.mul_add(v_elem, data));
                }
            } else if coe::is_same::<Item, f32>() {
                for (out, &data, &u_elem, &v_elem) in itertools::izip!(
                    output_data.as_mut_slice(),
                    chunk.data.as_slice(),
                    u_data.as_slice(),
                    v_data.as_slice()
                ) {
                    let data: f32 = coe::coerce_static(data);
                    let u_elem: f32 = coe::coerce_static(u_elem);
                    let v_elem: f32 = coe::coerce_static(v_elem);

                    *out = coe::coerce_static(u_elem.mul_add(v_elem, data));
                }
            } else if coe::is_same::<Item, c32>() {
                for (out, &data, &u_val, &v_val) in itertools::izip!(
                    output_data.as_mut_slice(),
                    chunk.data.as_slice(),
                    u_data.as_slice(),
                    v_data.as_slice()
                ) {
                    let data: c32 = coe::coerce_static(data);
                    let u_val: c32 = coe::coerce_static(u_val);
                    let v_val: c32 = coe::coerce_static(v_val);

                    *out = coe::coerce_static(c32::new(
                        data.re() + u_val.re().mul_add(v_val.re(), -u_val.im() * v_val.im()),
                        data.im() + u_val.re().mul_add(v_val.im(), u_val.im() * v_val.re()),
                    ));
                }
            } else if coe::is_same::<Item, c64>() {
                for (out, &data, &u_val, &v_val) in itertools::izip!(
                    output_data.as_mut_slice(),
                    chunk.data.as_slice(),
                    u_data.as_slice(),
                    v_data.as_slice()
                ) {
                    let data: c64 = coe::coerce_static(data);
                    let u_val: c64 = coe::coerce_static(u_val);
                    let v_val: c64 = coe::coerce_static(v_val);

                    *out = coe::coerce_static(c64::new(
                        data.re() + u_val.re().mul_add(v_val.re(), -u_val.im() * v_val.im()),
                        data.im() + u_val.re().mul_add(v_val.im(), u_val.im() * v_val.re()),
                    ));
                }
            } else {
                panic!("Unknown type");
            }
            Some(DataChunk::<Item, N> {
                data: output_data,
                start_index: chunk.start_index,
                valid_entries: chunk.valid_entries,
            })
        } else {
            None
        }
    }
}

impl<Item: RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>>
    Array<Item, ArrayImpl, 2>
{
    /// Return the rank-1 update object `self + uv^T`
    pub fn rank1_sum<
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    >(
        self,
        u: Array<Item, ArrayImplU, 1>,
        v: Array<Item, ArrayImplV, 1>,
    ) -> Array<Item, Rank1UpdateSum<Item, ArrayImpl, ArrayImplU, ArrayImplV>, 2> {
        Array::new(Rank1UpdateSum::new(self, u, v))
    }
}

/// Rank-1 Update of a two-dimensional array as product.
pub struct Rank1UpdateProduct<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>,
    ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
> {
    arr: Array<Item, ArrayImpl, 2>,
    u: Array<Item, ArrayImplU, 1>,
    v: Array<Item, ArrayImplV, 1>,
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    > Rank1UpdateProduct<Item, ArrayImpl, ArrayImplU, ArrayImplV>
{
    /// Initialize a rank 1 update of the form `arr + uv^T`.
    pub fn new(
        arr: Array<Item, ArrayImpl, 2>,
        u: Array<Item, ArrayImplU, 1>,
        v: Array<Item, ArrayImplV, 1>,
    ) -> Self {
        assert_eq!(arr.shape()[0], u.shape()[0]);
        assert_eq!(arr.shape()[1], v.shape()[0]);
        Self { arr, u, v }
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    > UnsafeRandomAccessByValue<2> for Rank1UpdateProduct<Item, ArrayImpl, ArrayImplU, ArrayImplV>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; 2]) -> Self::Item {
        if coe::is_same::<Item, f64>() {
            let u_val: f64 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: f64 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            let arr_val: f64 = coe::coerce_static(self.arr.get_value_unchecked(multi_index));
            coe::coerce_static(u_val * v_val * arr_val)
        } else if coe::is_same::<Item, f32>() {
            let u_val: f32 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: f32 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            let arr_val: f32 = coe::coerce_static(self.arr.get_value_unchecked(multi_index));
            coe::coerce_static(u_val * v_val * arr_val)
        } else if coe::is_same::<Item, c32>() {
            let u_val: c32 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: c32 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            let arr_val: c32 = coe::coerce_static(self.arr.get_value_unchecked(multi_index));
            let prod = c32::new(
                u_val.re().mul_add(v_val.re(), -u_val.im() * v_val.im()),
                u_val.re().mul_add(v_val.im(), u_val.im() * v_val.re()),
            );
            coe::coerce_static(c32::new(
                arr_val.re().mul_add(prod.re(), -arr_val.im() * prod.im()),
                arr_val.re().mul_add(prod.im(), arr_val.im() * prod.re()),
            ))
        } else if coe::is_same::<Item, c64>() {
            let u_val: c64 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: c64 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            let arr_val: c64 = coe::coerce_static(self.arr.get_value_unchecked(multi_index));
            let prod = c64::new(
                u_val.re().mul_add(v_val.re(), -u_val.im() * v_val.im()),
                u_val.re().mul_add(v_val.im(), u_val.im() * v_val.re()),
            );
            coe::coerce_static(c64::new(
                arr_val.re().mul_add(prod.re(), -arr_val.im() * prod.im()),
                arr_val.re().mul_add(prod.im(), arr_val.im() * prod.re()),
            ))
        } else {
            panic!("Unknown type");
        }
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    > Shape<2> for Rank1UpdateProduct<Item, ArrayImpl, ArrayImplU, ArrayImplV>
{
    fn shape(&self) -> [usize; 2] {
        self.arr.shape()
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + ChunkedAccess<N, Item = Item>,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        const N: usize,
    > ChunkedAccess<N> for Rank1UpdateProduct<Item, ArrayImpl, ArrayImplU, ArrayImplV>
{
    type Item = Item;

    fn get_chunk(&self, chunk_index: usize) -> Option<crate::DataChunk<Self::Item, N>> {
        if let Some(chunk) = self.arr.get_chunk(chunk_index) {
            let mut output_data = [<Item as num::Zero>::zero(); N];
            let mut u_data = [<Item as num::Zero>::zero(); N];
            let mut v_data = [<Item as num::Zero>::zero(); N];
            let shape = self.arr.shape();
            for (index, u_elem, v_elem) in
                itertools::izip!(0..chunk.valid_entries, u_data.iter_mut(), v_data.iter_mut(),)
            {
                let multi_index = convert_1d_nd_from_shape(index + chunk.start_index, shape);
                *u_elem = unsafe { self.u.get_value_unchecked([multi_index[0]]) };
                *v_elem = unsafe { self.v.get_value_unchecked([multi_index[1]]) };
            }

            if coe::is_same::<Item, f64>() {
                for (out, &data, &u_elem, &v_elem) in itertools::izip!(
                    output_data.as_mut_slice(),
                    chunk.data.as_slice(),
                    u_data.as_slice(),
                    v_data.as_slice()
                ) {
                    let data: f64 = coe::coerce_static(data);
                    let u_elem: f64 = coe::coerce_static(u_elem);
                    let v_elem: f64 = coe::coerce_static(v_elem);

                    *out = coe::coerce_static(u_elem * v_elem * data);
                }
            } else if coe::is_same::<Item, f32>() {
                for (out, &data, &u_elem, &v_elem) in itertools::izip!(
                    output_data.as_mut_slice(),
                    chunk.data.as_slice(),
                    u_data.as_slice(),
                    v_data.as_slice()
                ) {
                    let data: f32 = coe::coerce_static(data);
                    let u_elem: f32 = coe::coerce_static(u_elem);
                    let v_elem: f32 = coe::coerce_static(v_elem);

                    *out = coe::coerce_static(u_elem * v_elem * data);
                }
            } else if coe::is_same::<Item, c32>() {
                for (out, &data, &u_val, &v_val) in itertools::izip!(
                    output_data.as_mut_slice(),
                    chunk.data.as_slice(),
                    u_data.as_slice(),
                    v_data.as_slice()
                ) {
                    let data: c32 = coe::coerce_static(data);
                    let u_val: c32 = coe::coerce_static(u_val);
                    let v_val: c32 = coe::coerce_static(v_val);

                    let prod = c32::new(
                        u_val.re().mul_add(v_val.re(), -u_val.im() * v_val.im()),
                        u_val.re().mul_add(v_val.im(), u_val.im() * v_val.re()),
                    );
                    *out = coe::coerce_static(c32::new(
                        data.re().mul_add(prod.re(), -data.im() * prod.im()),
                        data.re().mul_add(prod.im(), data.im() * prod.re()),
                    ));
                }
            } else if coe::is_same::<Item, c64>() {
                for (out, &data, &u_val, &v_val) in itertools::izip!(
                    output_data.as_mut_slice(),
                    chunk.data.as_slice(),
                    u_data.as_slice(),
                    v_data.as_slice()
                ) {
                    let data: c64 = coe::coerce_static(data);
                    let u_val: c64 = coe::coerce_static(u_val);
                    let v_val: c64 = coe::coerce_static(v_val);

                    let prod = c64::new(
                        u_val.re().mul_add(v_val.re(), -u_val.im() * v_val.im()),
                        u_val.re().mul_add(v_val.im(), u_val.im() * v_val.re()),
                    );
                    *out = coe::coerce_static(c64::new(
                        data.re().mul_add(prod.re(), -data.im() * prod.im()),
                        data.re().mul_add(prod.im(), data.im() * prod.re()),
                    ));
                }
            } else {
                panic!("Unknown type");
            }
            Some(DataChunk::<Item, N> {
                data: output_data,
                start_index: chunk.start_index,
                valid_entries: chunk.valid_entries,
            })
        } else {
            None
        }
    }
}

impl<Item: RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>>
    Array<Item, ArrayImpl, 2>
{
    /// Return the rank-1 update object `self + uv^T`
    pub fn rank1_cmp_product<
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    >(
        self,
        u: Array<Item, ArrayImplU, 1>,
        v: Array<Item, ArrayImplV, 1>,
    ) -> Array<Item, Rank1UpdateProduct<Item, ArrayImpl, ArrayImplU, ArrayImplV>, 2> {
        Array::new(Rank1UpdateProduct::new(self, u, v))
    }
}

#[cfg(test)]
mod test {

    use rand::SeedableRng;

    use super::*;
    use crate::{assert_array_abs_diff_eq, assert_array_relative_eq, prelude::*};

    #[test]
    fn test_rank1_update_sum_f32() {
        let eps = 1E-5;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(f32, [m]);
        let mut v = rlst_dynamic_array1!(f32, [n]);

        let mut arr = rlst_dynamic_array2!(f32, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);
        arr.fill_from_equally_distributed(&mut rng);

        let mut actual_nonchunked = rlst_dynamic_array2!(f32, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(f32, [m, n]);

        let mut actual_inplace = rlst_dynamic_array2!(f32, [m, n]);

        let mut expected = rlst_dynamic_array2!(f32, [m, n]);

        expected.fill_from(arr.view());

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] += u[[row_index]] * v[[col_index]];
            }
        }

        actual_nonchunked.fill_from(arr.view().rank1_sum(u.view(), v.view()));

        actual_chunked.fill_from_chunked::<_, 16>(arr.view().rank1_sum(u.view(), v.view()));

        actual_inplace.fill_from(arr.view());
        actual_inplace.rank1_sum_inplace(u.view(), v.view());

        assert_array_relative_eq!(actual_nonchunked, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
        assert_array_relative_eq!(actual_inplace, expected, eps);
    }

    #[test]
    fn test_rank1_update_sum_f64() {
        let eps = 1E-12;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(f64, [m]);
        let mut v = rlst_dynamic_array1!(f64, [n]);

        let mut arr = rlst_dynamic_array2!(f64, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);
        arr.fill_from_equally_distributed(&mut rng);

        let mut actual_nonchunked = rlst_dynamic_array2!(f64, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(f64, [m, n]);

        let mut actual_inplace = rlst_dynamic_array2!(f64, [m, n]);

        let mut expected = rlst_dynamic_array2!(f64, [m, n]);

        expected.fill_from(arr.view());

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] += u[[row_index]] * v[[col_index]];
            }
        }

        actual_nonchunked.fill_from(arr.view().rank1_sum(u.view(), v.view()));

        actual_chunked.fill_from_chunked::<_, 16>(arr.view().rank1_sum(u.view(), v.view()));

        actual_inplace.fill_from(arr.view());
        actual_inplace.rank1_sum_inplace(u.view(), v.view());

        assert_array_relative_eq!(actual_nonchunked, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
        assert_array_relative_eq!(actual_inplace, expected, eps);
    }

    #[test]
    fn test_rank1_update_sum_c64() {
        let eps = 1E-12;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(c64, [m]);
        let mut v = rlst_dynamic_array1!(c64, [n]);

        let mut arr = rlst_dynamic_array2!(c64, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);
        arr.fill_from_equally_distributed(&mut rng);

        let mut actual_nonchunked = rlst_dynamic_array2!(c64, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(c64, [m, n]);

        let mut actual_inplace = rlst_dynamic_array2!(c64, [m, n]);

        let mut expected = rlst_dynamic_array2!(c64, [m, n]);

        expected.fill_from(arr.view());

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] += u[[row_index]] * v[[col_index]];
            }
        }

        actual_nonchunked.fill_from(arr.view().rank1_sum(u.view(), v.view()));

        actual_chunked.fill_from_chunked::<_, 16>(arr.view().rank1_sum(u.view(), v.view()));

        actual_inplace.fill_from(arr.view());
        actual_inplace.rank1_sum_inplace(u.view(), v.view());

        assert_array_relative_eq!(actual_nonchunked, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
        assert_array_relative_eq!(actual_inplace, expected, eps);
    }

    #[test]
    fn test_rank1_update_sum_c32() {
        let eps = 1E-5;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(c32, [m]);
        let mut v = rlst_dynamic_array1!(c32, [n]);

        let mut arr = rlst_dynamic_array2!(c32, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);
        arr.fill_from_equally_distributed(&mut rng);

        let mut actual_nonchunked = rlst_dynamic_array2!(c32, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(c32, [m, n]);

        let mut actual_inplace = rlst_dynamic_array2!(c32, [m, n]);

        let mut expected = rlst_dynamic_array2!(c32, [m, n]);

        expected.fill_from(arr.view());

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] += u[[row_index]] * v[[col_index]];
            }
        }

        actual_nonchunked.fill_from(arr.view().rank1_sum(u.view(), v.view()));

        actual_chunked.fill_from_chunked::<_, 16>(arr.view().rank1_sum(u.view(), v.view()));

        actual_inplace.fill_from(arr.view());
        actual_inplace.rank1_sum_inplace(u.view(), v.view());

        assert_array_relative_eq!(actual_nonchunked, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
        assert_array_relative_eq!(actual_inplace, expected, eps);
    }

    #[test]
    fn test_rank1_update_product_f32() {
        let eps = 1E-5;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(f32, [m]);
        let mut v = rlst_dynamic_array1!(f32, [n]);

        let mut arr = rlst_dynamic_array2!(f32, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);
        arr.fill_from_equally_distributed(&mut rng);

        let mut actual_nonchunked = rlst_dynamic_array2!(f32, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(f32, [m, n]);

        let mut actual_inplace = rlst_dynamic_array2!(f32, [m, n]);

        let mut expected = rlst_dynamic_array2!(f32, [m, n]);

        expected.fill_from(arr.view());

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] *= u[[row_index]] * v[[col_index]];
            }
        }

        actual_nonchunked.fill_from(arr.view().rank1_cmp_product(u.view(), v.view()));

        actual_chunked.fill_from_chunked::<_, 16>(arr.view().rank1_cmp_product(u.view(), v.view()));

        actual_inplace.fill_from(arr.view());
        actual_inplace.rank1_cmp_product_inplace(u.view(), v.view());

        assert_array_relative_eq!(actual_nonchunked, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
        assert_array_relative_eq!(actual_inplace, expected, eps);
    }

    #[test]
    fn test_rank1_update_product_f64() {
        let eps = 1E-12;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(f64, [m]);
        let mut v = rlst_dynamic_array1!(f64, [n]);

        let mut arr = rlst_dynamic_array2!(f64, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);
        arr.fill_from_equally_distributed(&mut rng);

        let mut actual_nonchunked = rlst_dynamic_array2!(f64, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(f64, [m, n]);

        let mut actual_inplace = rlst_dynamic_array2!(f64, [m, n]);

        let mut expected = rlst_dynamic_array2!(f64, [m, n]);

        expected.fill_from(arr.view());

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] *= u[[row_index]] * v[[col_index]];
            }
        }

        actual_nonchunked.fill_from(arr.view().rank1_cmp_product(u.view(), v.view()));

        actual_chunked.fill_from_chunked::<_, 16>(arr.view().rank1_cmp_product(u.view(), v.view()));

        actual_inplace.fill_from(arr.view());
        actual_inplace.rank1_cmp_product_inplace(u.view(), v.view());

        assert_array_relative_eq!(actual_nonchunked, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
        assert_array_relative_eq!(actual_inplace, expected, eps);
    }

    #[test]
    fn test_rank1_update_product_c64() {
        let eps = 1E-12;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(c64, [m]);
        let mut v = rlst_dynamic_array1!(c64, [n]);

        let mut arr = rlst_dynamic_array2!(c64, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);
        arr.fill_from_equally_distributed(&mut rng);

        let mut actual_nonchunked = rlst_dynamic_array2!(c64, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(c64, [m, n]);

        let mut actual_inplace = rlst_dynamic_array2!(c64, [m, n]);

        let mut expected = rlst_dynamic_array2!(c64, [m, n]);

        expected.fill_from(arr.view());

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] *= u[[row_index]] * v[[col_index]];
            }
        }

        actual_nonchunked.fill_from(arr.view().rank1_cmp_product(u.view(), v.view()));

        actual_chunked.fill_from_chunked::<_, 16>(arr.view().rank1_cmp_product(u.view(), v.view()));

        actual_inplace.fill_from(arr.view());
        actual_inplace.rank1_cmp_product_inplace(u.view(), v.view());

        assert_array_relative_eq!(actual_nonchunked, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
        assert_array_relative_eq!(actual_inplace, expected, eps);
    }

    #[test]
    fn test_rank1_update_product_c32() {
        let eps = 1E-5;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(c32, [m]);
        let mut v = rlst_dynamic_array1!(c32, [n]);

        let mut arr = rlst_dynamic_array2!(c32, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);
        arr.fill_from_equally_distributed(&mut rng);

        let mut actual_nonchunked = rlst_dynamic_array2!(c32, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(c32, [m, n]);

        let mut actual_inplace = rlst_dynamic_array2!(c32, [m, n]);

        let mut expected = rlst_dynamic_array2!(c32, [m, n]);

        expected.fill_from(arr.view());

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] *= u[[row_index]] * v[[col_index]];
            }
        }

        actual_nonchunked.fill_from(arr.view().rank1_cmp_product(u.view(), v.view()));

        actual_chunked.fill_from_chunked::<_, 16>(arr.view().rank1_cmp_product(u.view(), v.view()));

        actual_inplace.fill_from(arr.view());
        actual_inplace.rank1_cmp_product_inplace(u.view(), v.view());

        assert_array_relative_eq!(actual_nonchunked, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
        assert_array_relative_eq!(actual_inplace, expected, eps);
    }
}
