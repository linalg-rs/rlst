//! Example of a distributed CG iteration across several MPI ranks.
use rlst::operator::Operator;
use rlst::{rlst_dynamic_array2, zero_element, GmresIteration, OperatorBase, RawAccessMut};

pub fn main() {
    // We setup a diagonal sparse matrix on the first rank, send this
    // around and then wrap this in an operator to run CG.

    // The matrix dimension.
    let n = 5;
    let tol = 1E-5;

    let mut mat = rlst_dynamic_array2!(f64, [n, n]);
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
    mat.r_mut()[[4, 4]] = 3.0;

    let mut residuals = Vec::<f64>::new();
    // We can now wrap the matrix into an operator.
    let op = Operator::from(mat);
    // Let's create a right-hand side.
    let mut rhs = zero_element(op.range());
    rhs.view_mut().data_mut()[0] = 1.0;
    rhs.view_mut().data_mut()[1] = 2.0;
    rhs.view_mut().data_mut()[3] = 1.0;
    rhs.view_mut().data_mut()[4] = 3.0;
    // We need the vector x as well.
    // We can now run the GMRES iteration.
    let gmres = (GmresIteration::new(op.r(), rhs.r(), n))
        .set_callable(|_, res| {
            residuals.push(res);
        })
        .set_tol(tol)
        .set_restart(2);
    let (_sol, _res) = gmres.run();

    println!("Residuals: {:?}", residuals)
}
