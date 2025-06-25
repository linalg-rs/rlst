//! Operators on arrays

pub mod addition;
pub mod cast;
pub mod cmp_wise_division;
pub mod cmp_wise_product;
pub mod unary_op;
// pub mod other;
pub mod scalar_mult;
pub mod subtraction;
// pub mod to_complex;
pub mod reverse_axis;
pub mod transpose;

// pub fn simd_operator() {
//     let mut a = DynArray::<f32, 1>::from_shape([100]);
//     a.fill_from_seed_equally_distributed(0);
//
//     let b = a.apply_unary_op(|x| f32::sqrt(x)).eval();
//
//     println!("b[0] = {}", b[[0]]);
// }
