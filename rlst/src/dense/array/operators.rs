//! Operators on arrays

pub mod addition;
pub mod cast;
pub mod cmp_wise_division;
pub mod cmp_wise_product;
pub mod coerce;
pub mod mul_add;
pub mod negation;
pub mod reverse_axis;
pub mod scalar_mult;
pub mod subtraction;
pub mod transpose;
pub mod unary_op;
pub mod with_type_hint;

pub fn simd_operator() {
    use crate::DynArray;
    use crate::traits::EvaluateObject;
    use crate::traits::MulAdd;

    let mut a = DynArray::<f32, 2>::from_shape([5, 100]);
    let mut b = DynArray::<f32, 2>::from_shape([5, 100]);

    a.fill_from_seed_equally_distributed(1);
    b.fill_from_seed_equally_distributed(2);

    let c = (5.0f32 * a + 2.0f32 * b.sqrt()).eval();

    println!("c: {:?}", c.get_value([0, 0]).unwrap());
}
