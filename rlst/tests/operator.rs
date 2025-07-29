//! Tests for the operator interface

// use num::traits::{One, Zero};
// use rand::Rng;
// use rlst::{
//     operator::{zero_element, Operator},
//     prelude::*,
// };

// #[test]
// fn test_dense_matrix_operator() {
//     let mut mat = rlst_dynamic_array2!(f64, [3, 4]);
//     mat.fill_from_seed_equally_distributed(0);

//     let op = Operator::from(mat);
//     let mut x = zero_element(op.domain());
//     let mut y = zero_element(op.range());

//     x.view_mut().fill_from_seed_equally_distributed(0);

//     op.apply_extended(1.0, x.r(), 0.0, y.r_mut());
// }

// #[test]
// pub fn test_gram_schmidt() {
//     let space = ArrayVectorSpace::<c64>::from_dimension(5);
//     let mut vec1 = zero_element(space.clone());
//     let mut vec2 = zero_element(space.clone());
//     let mut vec3 = zero_element(space.clone());

//     vec1.view_mut().fill_from_seed_equally_distributed(0);
//     vec2.view_mut().fill_from_seed_equally_distributed(1);
//     vec3.view_mut().fill_from_seed_equally_distributed(2);

//     let mut frame = VectorFrame::default();

//     let mut original = VectorFrame::default();

//     frame.push(vec1);
//     frame.push(vec2);
//     frame.push(vec3);

//     for elem in frame.iter() {
//         original.push(elem.duplicate());
//     }

//     let mut r_mat = rlst_dynamic_array2!(c64, [3, 3]);

//     ModifiedGramSchmidt::orthogonalize(&mut frame, &mut r_mat);

//     // Check orthogonality
//     for index1 in 0..3 {
//         for index2 in 0..3 {
//             let inner = frame
//                 .get(index1)
//                 .unwrap()
//                 .inner_product(frame.get(index2).unwrap().r());
//             if index1 == index2 {
//                 approx::assert_relative_eq!(inner, c64::one(), epsilon = 1E-12);
//             } else {
//                 approx::assert_abs_diff_eq!(inner, c64::zero(), epsilon = 1E-12);
//             }
//         }
//     }

//     // Check that r is correct.
//     for (index, col) in r_mat.col_iter().enumerate() {
//         let mut actual = <ArrayVectorSpace<c64> as LinearSpace>::zero(space.clone());
//         let expected = original.get(index).unwrap();
//         let mut coeffs = rlst_dynamic_array1!(c64, [frame.len()]);
//         coeffs.fill_from(col.r());
//         frame.evaluate(coeffs.data(), actual.r_mut());
//         let rel_diff = (actual.view() - expected.view()).norm_2() / expected.view().norm_2();
//         approx::assert_abs_diff_eq!(rel_diff, f64::zero(), epsilon = 1E-12);
//     }
// }

// #[test]
// fn test_cg() {
//     let dim = 10;
//     let tol = 1E-5;

//     let mut residuals = Vec::<f64>::new();

//     let mut rng = rand::thread_rng();

//     let mut mat = rlst_dynamic_array2!(f64, [dim, dim]);

//     for index in 0..dim {
//         mat[[index, index]] = rng.gen_range(1.0..=2.0);
//     }

//     let op = Operator::from(mat.r());

//     let mut rhs = zero_element(op.range());
//     rhs.view_mut().fill_from_equally_distributed(&mut rng);

//     let cg = (CgIteration::new(op.r(), rhs.r()))
//         .set_callable(|_, res| {
//             let res_norm = res.norm();
//             residuals.push(res_norm);
//         })
//         .set_tol(tol)
//         .print_debug();

//     let (_sol, res) = cg.run();
//     assert!(res < tol);
// }

// #[test]
// fn test_operator_algebra() {
//     let mut mat1 = rlst_dynamic_array2!(f64, [4, 3]);
//     let mut mat2 = rlst_dynamic_array2!(f64, [4, 3]);

//     let domain = ArrayVectorSpace::from_dimension(3);
//     let range = ArrayVectorSpace::from_dimension(4);

//     mat1.fill_from_seed_equally_distributed(0);
//     mat2.fill_from_seed_equally_distributed(1);

//     let op1 = Operator::from(mat1);
//     let op2 = Operator::from(mat2);

//     let mut x = ArrayVectorSpace::zero(domain.clone());
//     let mut y = ArrayVectorSpace::zero(range.clone());
//     let mut y_expected = ArrayVectorSpace::zero(range.clone());
//     x.view_mut().fill_from_seed_equally_distributed(2);
//     y.view_mut().fill_from_seed_equally_distributed(3);
//     y_expected.view_mut().fill_from(y.view());

//     op2.apply_extended(2.0, x.r(), 3.5, y_expected.r_mut());
//     op1.apply_extended(10.0, x.r(), 1.0, y_expected.r_mut());

//     let sum = op1.scale(5.0).sum(op2.r());

//     sum.apply_extended(2.0, x.r(), 3.5, y.r_mut());

//     rlst::assert_array_relative_eq!(y.view(), y_expected.view(), 1E-12);
// }
