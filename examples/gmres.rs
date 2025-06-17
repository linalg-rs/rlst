//! Example of GMRES iterations
use num::Complex;
use rlst::operator::Operator;
use rlst::{c64, rlst_dynamic_array2, zero_element, GmresIteration, OperatorBase, RawAccessMut};

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

    // Wrap matrix into operator
    let op = Operator::from(mat);

    // Right-hand side
    let mut rhs = zero_element(op.range());
    rhs.view_mut().data_mut()[0] = 1.0;
    rhs.view_mut().data_mut()[1] = 2.0;
    rhs.view_mut().data_mut()[2] = 3.0;
    rhs.view_mut().data_mut()[3] = 4.0;
    rhs.view_mut().data_mut()[4] = 5.0;

    let mut residuals = Vec::<f64>::new();
    let tol = 1e-3;

    // GMRES solve
    let gmres = (GmresIteration::new(op.r(), rhs.r(), 5))
        .set_callable(|_, res| {
            residuals.push(res);
        })
        .set_tol(tol)
        .set_restart(5);

    let (_sol, _res) = gmres.run();

    println!("Residuals: {:?}", residuals);
    // As an example, we set up a complex matrix and
    // hen wrap it in an operator to run GMRES.
    let mut mat = rlst_dynamic_array2!(c64, [5, 5]);
    mat.r_mut()[[0, 0]] = Complex::new(4.0, 1.0);
    mat.r_mut()[[0, 1]] = Complex::new(1.0, -1.0);

    mat.r_mut()[[1, 0]] = Complex::new(1.0, 1.0);
    mat.r_mut()[[1, 1]] = Complex::new(4.0, 2.0);
    mat.r_mut()[[1, 2]] = Complex::new(1.0, -1.0);

    mat.r_mut()[[2, 1]] = Complex::new(1.0, 1.0);
    mat.r_mut()[[2, 2]] = Complex::new(4.0, 3.0);
    mat.r_mut()[[2, 3]] = Complex::new(1.0, -1.0);

    mat.r_mut()[[3, 2]] = Complex::new(1.0, 1.0);
    mat.r_mut()[[3, 3]] = Complex::new(4.0, 4.0);
    mat.r_mut()[[3, 4]] = Complex::new(1.0, -1.0);

    mat.r_mut()[[4, 3]] = Complex::new(1.0, 1.0);
    mat.r_mut()[[4, 4]] = Complex::new(4.0, 5.0);

    // We can now wrap the matrix into an operator.
    let op = Operator::from(mat);

    // Let's create a right-hand side.
    let mut rhs = zero_element(op.range());
    rhs.view_mut().data_mut()[0] = Complex::new(1.0, 2.0);
    rhs.view_mut().data_mut()[1] = Complex::new(2.0, 3.0);
    rhs.view_mut().data_mut()[2] = Complex::new(3.0, 4.0);
    rhs.view_mut().data_mut()[3] = Complex::new(4.0, 5.0);
    rhs.view_mut().data_mut()[4] = Complex::new(5.0, 6.0);

    let mut residuals = Vec::<f64>::new();
    let tol = 1E-3;

    // We can now run the GMRES iteration. We must specify the dimension in this case
    let gmres = (GmresIteration::new(op.r(), rhs.r(), 5))
        .set_callable(|_, res| {
            residuals.push(res);
        })
        .set_tol(tol)
        .set_restart(5);
    let (_sol, _res) = gmres.run();

    println!("Residuals: {:?}", residuals)
}
