//! Example of GMRES iterations
use num::Complex;
use rlst::operator::operations::eigenvalues::eigs::Which;
use rlst::operator::Operator;
use rlst::{c64, rlst_dynamic_array2, Eigs};

pub fn main() {
    let mut mat = rlst_dynamic_array2!(f64, [5, 5]);
    mat.r_mut()[[0, 0]] = 4.0;
    mat.r_mut()[[0, 1]] = 1.0;

    mat.r_mut()[[1, 0]] = 1.0;
    mat.r_mut()[[1, 1]] = 4.0;
    mat.r_mut()[[1, 2]] = 1.0;

    mat.r_mut()[[2, 1]] = 1.0;
    mat.r_mut()[[2, 2]] = 4.0;
    mat.r_mut()[[2, 3]] = 1.0;

    mat.r_mut()[[3, 2]] = 1.0;
    mat.r_mut()[[3, 3]] = 4.0;
    mat.r_mut()[[3, 4]] = 1.0;

    mat.r_mut()[[4, 3]] = 1.0;
    mat.r_mut()[[4, 4]] = 4.0;

    // We can now wrap the matrix into an operator.
    let op = Operator::from(mat);

    let k = 3;
    let mut eigs = Eigs::new(op, 1e-10, None, None, Some(Which::LM));

    let (res1, _res2) = eigs.run(None, k, None, false);

    println!("Eigenvalues: {:?}", res1);

    // We can do the same for a complex case
    let mut mat = rlst_dynamic_array2!(c64, [4, 4]);
    mat.r_mut()[[0, 0]] = Complex::new(1.0, 2.0);
    mat.r_mut()[[0, 1]] = Complex::new(2.0, -1.0);
    mat.r_mut()[[0, 2]] = Complex::new(0.0, 0.0);
    mat.r_mut()[[0, 3]] = Complex::new(1.0, 0.0);

    mat.r_mut()[[1, 0]] = Complex::new(0.0, 0.0);
    mat.r_mut()[[1, 1]] = Complex::new(3.0, 0.0);
    mat.r_mut()[[1, 2]] = Complex::new(1.0, 1.0);
    mat.r_mut()[[1, 3]] = Complex::new(0.0, 0.0);

    mat.r_mut()[[2, 0]] = Complex::new(4.0, 0.0);
    mat.r_mut()[[2, 1]] = Complex::new(0.0, 0.0);
    mat.r_mut()[[2, 2]] = Complex::new(2.0, -1.0);
    mat.r_mut()[[2, 3]] = Complex::new(1.0, 0.0);

    mat.r_mut()[[3, 0]] = Complex::new(0.0, 1.0);
    mat.r_mut()[[3, 1]] = Complex::new(0.0, 0.0);
    mat.r_mut()[[3, 2]] = Complex::new(0.0, 0.0);
    mat.r_mut()[[3, 3]] = Complex::new(1.0, 3.0);

    let op = Operator::from(mat);

    let k = 2;
    let mut eigs = Eigs::new(op, 1e-10, None, None, Some(Which::LM));

    let (res1, _res2) = eigs.run(None, k, None, false);

    println!("Eigenvalues: {:?}", res1);
}
