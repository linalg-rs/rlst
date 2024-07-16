//! Various unary Maths operators

use coe::Coerce;
use paste::paste;

use crate::{
    dense::{
        array::{Array, ChunkedAccess, DataChunk, Shape, UnsafeRandomAccessByValue},
        types::RlstScalar,
    },
    RlstSimd, SimdFor,
};

macro_rules! impl_unary_op {
    ($fun:expr, $name:expr) => {
        paste! {

        #[doc = "Implements "]
        #[doc = stringify!($name)]
        #[doc = " operator."]
        pub struct [<Array $name >]<
            Item: RlstScalar,
            ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
            const NDIM: usize,
        > {
            operator: Array<Item, ArrayImpl, NDIM>,
        }

        impl<
                Item: RlstScalar,
                ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
                const NDIM: usize,
            > [<Array $name>]<Item, ArrayImpl, NDIM>
        {
            /// Create new
            pub fn new(operator: Array<Item, ArrayImpl, NDIM>) -> Self {
                Self { operator }
            }
        }

        impl<
                Item: RlstScalar,
                ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
                const NDIM: usize,
            > UnsafeRandomAccessByValue<NDIM> for [<Array $name>]<Item, ArrayImpl, NDIM>
        {
            type Item = Item;
            #[inline]
            unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
                self.operator.get_value_unchecked(multi_index).[<$fun>]()
            }
        }

        impl<
                Item: RlstScalar,
                ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
                const NDIM: usize,
            > Shape<NDIM> for [<Array $name>]<Item, ArrayImpl, NDIM>
        {
            fn shape(&self) -> [usize; NDIM] {
                self.operator.shape()
            }
        }

        impl<
                Item: RlstScalar,
                ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
                const NDIM: usize,
            > Array<Item, ArrayImpl, NDIM>
        {
            #[doc = "Implements "]
            #[doc = stringify!($name)]
            #[doc = " operator."]
            pub fn [<$fun>](self) -> Array<Item, [<Array $name >]<Item, ArrayImpl, NDIM>, NDIM> {
                Array::new([<Array $name>]::new(self))
            }
        }

        impl<
                Item: RlstScalar,
                ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
                    + Shape<NDIM>
                    + ChunkedAccess<N, Item = Item>,
                const NDIM: usize,
                const N: usize,
            > ChunkedAccess<N> for [<Array $name>]<Item, ArrayImpl, NDIM>
        {
            type Item = Item;
            #[inline]
            fn get_chunk(&self, chunk_index: usize) -> Option<DataChunk<Self::Item, N>> {
                if let Some(mut chunk) = self.operator.get_chunk(chunk_index) {
                    struct Impl<'a, Item: RlstSimd> {
                        data: &'a mut [Item],
                    }

                    impl<'a, Item: RlstSimd> pulp::WithSimd for Impl<'a, Item> {
                        type Output = ();

                        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                            let Self { data } = self;
                            let simd = SimdFor::<Item, S>::new(simd);
                            let (arr_head, arr_tail) =
                                <Item as RlstSimd>::as_simd_slice_mut::<S>(data);
                            for s in arr_head {
                                *s = simd.[<$fun>](*s);
                            }
                            for s in arr_tail {
                                *s = s.[<$fun>]();
                            }
                        }
                    }

                    if coe::is_same::<Item, f32>() {
                        pulp::Arch::new().dispatch(Impl::<'_, f32> {
                            data: chunk.data.as_mut_slice().coerce(),
                        });
                    } else if coe::is_same::<Item, f64>() {
                        pulp::Arch::new().dispatch(Impl::<'_, f64> {
                            data: chunk.data.as_mut_slice().coerce(),
                        });
                    } else {
                        for item in &mut chunk.data {
                            *item = item.[<$fun>]();
                        }
                    }
                    Some(chunk)
                } else {
                    None
                }
            }
        }
        }
    };
}

impl_unary_op!(exp, Exp);
impl_unary_op!(sqrt, Sqrt);

#[cfg(test)]
mod test {

    use crate::{assert_array_relative_eq, prelude::*};
    use paste::paste;

    macro_rules! impl_unary_op_test {
        ($fun:expr, $scalar:ty, $tol:expr) => {
            paste! {
            #[test]
            fn [<test_array_ $fun _ $scalar>]() {
                let n = 53;

                let mut arr = rlst_dynamic_array1!($scalar, [n]);
                arr.fill_from_seed_equally_distributed(0);

                let mut actual = rlst_dynamic_array1!($scalar, [n]);
                let mut expected = rlst_dynamic_array1!($scalar, [n]);

                for (ar, e) in itertools::izip!(arr.iter(), expected.iter_mut()) {
                    *e = ar.exp();
                }

                actual.fill_from(arr.view().exp());

                assert_array_relative_eq!(actual, expected, $tol);

                actual.set_zero();
                actual.fill_from_chunked::<_, 19>(arr.view().exp());

                assert_array_relative_eq!(actual, expected, $tol);
            }

            }
        };
    }

    impl_unary_op_test!(sqrt, f32, 1E-5);
    impl_unary_op_test!(exp, f32, 1E-5);

    impl_unary_op_test!(sqrt, f64, 1E-10);
    impl_unary_op_test!(exp, f64, 1E-10);
}
