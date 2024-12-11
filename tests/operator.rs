//! Tests for the operator interface

use num::traits::{One, Zero};
use rand::Rng;
use rlst::prelude::*;

#[test]
fn test_dense_matrix_operator() {
    let mut mat = rlst_dynamic_array2!(f64, [3, 4]);
    let domain = ArrayVectorSpace::new(4);
    let range = ArrayVectorSpace::new(3);
    mat.fill_from_seed_equally_distributed(0);

    let op = DenseMatrixOperator::new(mat, &domain, &range);
    let mut x = op.domain().zero();
    let mut y = op.range().zero();

    x.view_mut().fill_from_seed_equally_distributed(0);

    op.apply_extended(1.0, &x, 0.0, &mut y).unwrap();
}

#[test]
pub fn test_gram_schmidt() {
    let space = ArrayVectorSpace::<c64>::new(5);
    let mut vec1 = space.zero();
    let mut vec2 = space.zero();
    let mut vec3 = space.zero();

    vec1.view_mut().fill_from_seed_equally_distributed(0);
    vec2.view_mut().fill_from_seed_equally_distributed(1);
    vec3.view_mut().fill_from_seed_equally_distributed(2);

    let mut frame = VectorFrame::default();

    let mut original = VectorFrame::default();

    frame.push(vec1);
    frame.push(vec2);
    frame.push(vec3);

    for elem in frame.iter() {
        original.push(space.new_from(elem));
    }

    let mut r_mat = rlst_dynamic_array2!(c64, [3, 3]);

    ModifiedGramSchmidt::orthogonalize(&space, &mut frame, &mut r_mat);

    // Check orthogonality
    for index1 in 0..3 {
        for index2 in 0..3 {
            let inner = space.inner(frame.get(index1).unwrap(), frame.get(index2).unwrap());
            if index1 == index2 {
                approx::assert_relative_eq!(inner, c64::one(), epsilon = 1E-12);
            } else {
                approx::assert_abs_diff_eq!(inner, c64::zero(), epsilon = 1E-12);
            }
        }
    }

    // Check that r is correct.
    for (index, col) in r_mat.col_iter().enumerate() {
        let mut actual = space.zero();
        let expected = original.get(index).unwrap();
        let mut coeffs = rlst_dynamic_array1!(c64, [frame.len()]);
        coeffs.fill_from(col.r());
        frame.evaluate(coeffs.data(), &mut actual);
        let rel_diff = (actual.view() - expected.view()).norm_2() / expected.view().norm_2();
        approx::assert_abs_diff_eq!(rel_diff, f64::zero(), epsilon = 1E-12);
    }
}

#[test]
fn test_cg() {
    let dim = 10;
    let tol = 1E-5;

    let space = ArrayVectorSpace::<f64>::new(dim);
    let mut residuals = Vec::<f64>::new();

    let mut rng = rand::thread_rng();

    let mut mat = rlst_dynamic_array2!(f64, [dim, dim]);

    for index in 0..dim {
        mat[[index, index]] = rng.gen_range(1.0..=2.0);
    }

    let op = DenseMatrixOperator::new(mat.r(), &space, &space);

    let mut rhs = space.zero();
    rhs.view_mut().fill_from_equally_distributed(&mut rng);

    let cg = (CgIteration::new(&op, &rhs))
        .set_callable(|_, res| {
            let res_norm = space.norm(res);
            residuals.push(res_norm);
        })
        .set_tol(tol)
        .print_debug();

    let (_sol, res) = cg.run();
    assert!(res < tol);
}

#[test]
fn test_operator_algebra() {
    let mut mat1 = rlst_dynamic_array2!(f64, [4, 3]);
    let mut mat2 = rlst_dynamic_array2!(f64, [4, 3]);

    let domain = ArrayVectorSpace::new(3);
    let range = ArrayVectorSpace::new(4);

    mat1.fill_from_seed_equally_distributed(0);
    mat2.fill_from_seed_equally_distributed(1);

    let op1 = DenseMatrixOperator::new(mat1, &domain, &range);
    let op2 = DenseMatrixOperator::new(mat2, &domain, &range);

    let mut x = domain.zero();
    let mut y = range.zero();
    let mut y_expected = range.zero();
    x.view_mut().fill_from_seed_equally_distributed(2);
    y.view_mut().fill_from_seed_equally_distributed(3);
    y_expected.view_mut().fill_from(y.view());

    op2.apply_extended(2.0, &x, 3.5, &mut y_expected).unwrap();
    op1.apply_extended(10.0, &x, 1.0, &mut y_expected).unwrap();

    let sum = op1.scale(5.0).sum(op2.as_ref_obj());

    sum.apply_extended(2.0, &x, 3.5, &mut y).unwrap();

    rlst::assert_array_relative_eq!(y.view(), y_expected.view(), 1E-12);
}
