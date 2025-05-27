//! Example of a distributed CG iteration across several MPI ranks.
use num::Complex;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rlst::operator::Operator;
use rlst::{rlst_dynamic_array2, zero_element, GmresIteration, OperatorBase, RawAccessMut};

pub fn main() {
  

    // We setup a diagonal sparse matrix on the first rank, send this
    // around and then wrap this in an operator to run CG.

    // The matrix dimension.
    let n = 30;
    let tol = 1E-5;

    let mut rng1 = StdRng::seed_from_u64(0);
    let mut rng2 = StdRng::seed_from_u64(1);
    let mut mat = rlst_dynamic_array2!(Complex<f64>, [n, n]);

    mat.fill_from_equally_distributed(&mut rng1);

    let mut residuals = Vec::<f64>::new();
    // We can now wrap the matrix into an operator.
    let op = Operator::from(mat);
    // Let's create a right-hand side.
    let mut rhs = zero_element(op.range());
    for val in rhs.view_mut().data_mut() {
        let re = rand::Rng::gen::<f64>(&mut rng2);
        let im = rand::Rng::gen::<f64>(&mut rng2);
        *val = Complex::new(re, im);
    }

    // We need the vector x as well.
    // We can now run the GMRES iteration.
    let gmres = (GmresIteration::new(op.r(), rhs.r(), n))
        .set_callable(|_, res| {
            let res_norm = res.norm();
            residuals.push(res_norm);
        })
        .set_tol(tol);
    let (_sol, res) = gmres.run();

    println!("The residual is: {}", res);
    
}
