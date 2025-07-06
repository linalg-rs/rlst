//! Operators on arrays

pub mod addition;
pub mod cast;
pub mod cmp_wise_division;
pub mod cmp_wise_product;
pub mod coerce;
pub mod mul_add;
pub mod reverse_axis;
pub mod scalar_mult;
pub mod subtraction;
pub mod transpose;
pub mod unary_op;
pub mod with_type_hint;

// pub fn simd_operator() {
//     let mut a = DynArray::<f32, 2>::from_shape([5, 100]);
//     let mut b = DynArray::<f32, 2>::from_shape([5, 100]);
//
//     a.fill_from_seed_normally_distributed(1);
//     b.fill_from_seed_normally_distributed(2);
//
//     let c = a.mul_add(5.0f32, 2.0f32 * b.sin()).eval();
//
//     println!("c: {:?}", RandomAccessByValue::get_value(&c, [0, 0]));
// }
