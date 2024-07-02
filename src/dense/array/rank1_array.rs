//! A rank1 update of a two-dimensional array.

use crate::dense::layout::convert_1d_nd_from_shape;
use crate::dense::types::{c32, c64};

use crate::{Array, ChunkedAccess, DataChunk, RlstScalar, Shape, UnsafeRandomAccessByValue};

use super::empty_chunk;

/// Rank-1 Update of a two-dimensional array.
pub struct Rank1Array<
    Item: RlstScalar,
    ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
> {
    u: Array<Item, ArrayImplU, 1>,
    v: Array<Item, ArrayImplV, 1>,
}

impl<
        Item: RlstScalar,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    > Rank1Array<Item, ArrayImplU, ArrayImplV>
{
    /// Initialize a rank 1 update of the form `arr + uv^T`.
    pub fn new(u: Array<Item, ArrayImplU, 1>, v: Array<Item, ArrayImplV, 1>) -> Self {
        Self { u, v }
    }
}

impl<
        Item: RlstScalar,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    > UnsafeRandomAccessByValue<2> for Rank1Array<Item, ArrayImplU, ArrayImplV>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; 2]) -> Self::Item {
        if coe::is_same::<Item, f64>() {
            let u_val: f64 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: f64 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            coe::coerce_static(u_val * v_val)
        } else if coe::is_same::<Item, f32>() {
            let u_val: f32 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: f32 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            coe::coerce_static(u_val * v_val)
        } else if coe::is_same::<Item, c32>() {
            let u_val: c32 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: c32 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            coe::coerce_static(c32::new(
                u_val.re().mul_add(v_val.re(), -u_val.im() * v_val.im()),
                u_val.re().mul_add(v_val.im(), u_val.im() * v_val.re()),
            ))
        } else if coe::is_same::<Item, c64>() {
            let u_val: c64 = coe::coerce_static(self.u.get_value_unchecked([multi_index[0]]));
            let v_val: c64 = coe::coerce_static(self.v.get_value_unchecked([multi_index[1]]));
            coe::coerce_static(c64::new(
                u_val.re().mul_add(v_val.re(), -u_val.im() * v_val.im()),
                u_val.re().mul_add(v_val.im(), u_val.im() * v_val.re()),
            ))
        } else {
            panic!("Unknown type");
        }
    }
}

impl<
        Item: RlstScalar,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    > Shape<2> for Rank1Array<Item, ArrayImplU, ArrayImplV>
{
    fn shape(&self) -> [usize; 2] {
        [self.u.shape()[0], self.v.shape()[0]]
    }
}

impl<
        Item: RlstScalar,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        const N: usize,
    > ChunkedAccess<N> for Rank1Array<Item, ArrayImplU, ArrayImplV>
{
    type Item = Item;

    fn get_chunk(&self, chunk_index: usize) -> Option<crate::DataChunk<Self::Item, N>> {
        let nelements = self.u.shape()[0] * self.v.shape()[0];
        if let Some(chunk) = empty_chunk::<N, Item>(chunk_index, nelements) {
            let mut output_data = [<Item as num::Zero>::zero(); N];
            let mut u_data = [<Item as num::Zero>::zero(); N];
            let mut v_data = [<Item as num::Zero>::zero(); N];
            let shape = self.shape();
            for (index, u_elem, v_elem) in
                itertools::izip!(0..chunk.valid_entries, u_data.iter_mut(), v_data.iter_mut(),)
            {
                let multi_index = convert_1d_nd_from_shape(index + chunk.start_index, shape);
                *u_elem = unsafe { self.u.get_value_unchecked([multi_index[0]]) };
                *v_elem = unsafe { self.v.get_value_unchecked([multi_index[1]]) };
            }

            for (out, &u_elem, &v_elem) in itertools::izip!(
                output_data.as_mut_slice(),
                u_data.as_slice(),
                v_data.as_slice()
            ) {
                *out = u_elem * v_elem;
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

impl<
        Item: RlstScalar,
        ArrayImplU: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplV: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    > Rank1Array<Item, ArrayImplU, ArrayImplV>
{
    /// Return a new rank1 array that implements `u x v^T`
    pub fn new_array(
        u: Array<Item, ArrayImplU, 1>,
        v: Array<Item, ArrayImplV, 1>,
    ) -> Array<Item, Rank1Array<Item, ArrayImplU, ArrayImplV>, 2> {
        Array::new(Rank1Array::new(u, v))
    }
}

#[cfg(test)]
mod test {

    use approx::assert_relative_eq;
    use rand::SeedableRng;

    use crate::{assert_array_relative_eq, prelude::*};

    #[test]
    fn test_rank1_f32() {
        let eps = 1E-5;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(f32, [m]);
        let mut v = rlst_dynamic_array1!(f32, [n]);

        let mut expected = rlst_dynamic_array2!(f32, [m, n]);
        let mut actual = rlst_dynamic_array2!(f32, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(f32, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] = u[[row_index]] * v[[col_index]]
            }
        }

        let rank1_array = rlst_rank1_array!(u.view(), v.view());

        actual.fill_from(rank1_array.view());
        actual_chunked.fill_from_chunked::<_, 13>(rank1_array.view());

        assert_array_relative_eq!(rank1_array, expected, eps);
        assert_array_relative_eq!(actual, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
    }

    #[test]
    fn test_rank1_f64() {
        let eps = 1E-14;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(f64, [m]);
        let mut v = rlst_dynamic_array1!(f64, [n]);

        let mut expected = rlst_dynamic_array2!(f64, [m, n]);
        let mut actual = rlst_dynamic_array2!(f64, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(f64, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] = u[[row_index]] * v[[col_index]]
            }
        }

        let rank1_array = rlst_rank1_array!(u.view(), v.view());

        actual.fill_from(rank1_array.view());
        actual_chunked.fill_from_chunked::<_, 13>(rank1_array.view());

        assert_array_relative_eq!(rank1_array, expected, eps);
        assert_array_relative_eq!(actual, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
    }

    #[test]
    fn test_rank1_c32() {
        let eps = 1E-5;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(c32, [m]);
        let mut v = rlst_dynamic_array1!(c32, [n]);

        let mut expected = rlst_dynamic_array2!(c32, [m, n]);
        let mut actual = rlst_dynamic_array2!(c32, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(c32, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] = u[[row_index]] * v[[col_index]]
            }
        }

        let rank1_array = rlst_rank1_array!(u.view(), v.view());

        actual.fill_from(rank1_array.view());
        actual_chunked.fill_from_chunked::<_, 13>(rank1_array.view());

        assert_array_relative_eq!(rank1_array, expected, eps);
        assert_array_relative_eq!(actual, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
    }

    #[test]
    fn test_rank1_c64() {
        let eps = 1E-14;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let m = 50;
        let n = 25;

        let mut u = rlst_dynamic_array1!(c64, [m]);
        let mut v = rlst_dynamic_array1!(c64, [n]);

        let mut expected = rlst_dynamic_array2!(c64, [m, n]);
        let mut actual = rlst_dynamic_array2!(c64, [m, n]);
        let mut actual_chunked = rlst_dynamic_array2!(c64, [m, n]);

        u.fill_from_equally_distributed(&mut rng);
        v.fill_from_equally_distributed(&mut rng);

        for row_index in 0..m {
            for col_index in 0..n {
                expected[[row_index, col_index]] = u[[row_index]] * v[[col_index]]
            }
        }

        let rank1_array = rlst_rank1_array!(u.view(), v.view());

        actual.fill_from(rank1_array.view());
        actual_chunked.fill_from_chunked::<_, 13>(rank1_array.view());

        assert_array_relative_eq!(rank1_array, expected, eps);
        assert_array_relative_eq!(actual, expected, eps);
        assert_array_relative_eq!(actual_chunked, expected, eps);
    }
}
