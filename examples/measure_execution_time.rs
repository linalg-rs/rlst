//! Examples for measuring execution time

use rlst::{
    prelude::*,
    tracing::{trace_call, Tracing},
};

#[measure_duration(id = "matmul_fun")]
fn matmul_fun<
    ArrayImpl1: RandomAccessByValue<2, Item = f64> + Shape<2> + Stride<2> + RawAccess<Item = f64>,
    ArrayImpl2: RandomAccessByValue<2, Item = f64> + Shape<2> + Stride<2> + RawAccess<Item = f64>,
>(
    arr1: Array<f64, ArrayImpl1, 2>,
    arr2: Array<f64, ArrayImpl2, 2>,
) -> DynamicArray<f64, 2> {
    let res = empty_array().simple_mult_into_resize(arr1.r(), arr2.r());
    res
}

fn main() {
    let n = 5000;

    env_logger::init();

    let mut mat_a = rlst_dynamic_array2!(f64, [n, n]);
    let mut mat_b = rlst_dynamic_array2!(f64, [n, n]);

    mat_a.fill_from_seed_equally_distributed(0);
    mat_b.fill_from_seed_equally_distributed(1);

    let _mat_c = trace_call("mat_mul", || {
        empty_array().simple_mult_into_resize(mat_a.r(), mat_b.r())
    });

    let _mat_c2 = matmul_fun(mat_a.r(), mat_b.r());
}
