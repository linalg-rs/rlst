//! In this example we demonstrate the eigenvalue decomposition for nonsymmtric matrices

extern crate blas_src;
extern crate lapack_src;

use rlst::dense::linalg::lapack::eigenvalue_decomposition::EigMode;
use rlst::{DynArray, EigenvalueDecomposition, EvaluateObject, dot};
use rlst::{Inverse, c64};

fn main() {
    // Let's create a simple square matrix

    let n = 11;
    let mut a = DynArray::<f64, 2>::from_shape([n, n]);
    a.fill_from_seed_equally_distributed(0);

    // We can easily compute the eigenvalues of the matrix.

    let lam = a.eigenvalues().unwrap();

    // To compute the Schur decomposition we use the following command.

    let (t, z) = a.schur().unwrap();

    // Test the Schur decomposition

    let actual = dot!(z.r(), t.r(), z.r().conj().transpose().eval());

    rlst::assert_array_relative_eq!(actual, a, 1E-10);

    // We can also compute the full eigenvalue decomposition

    let (lam2, vr, vl) = a.eig(EigMode::BothEigenvectors).unwrap();

    rlst::assert_array_relative_eq!(lam, lam2, 1E-10);

    // Test the left eigenvectors

    // First convert a to a complex matrix
    let a_complex = a.into_type::<c64>().eval();

    // Now create a diagonal matrix from the eigenvalues

    let diag = rlst::diag!(lam2);

    // Now check the left eigenvectors

    let vlh = vl.unwrap().conj().transpose().eval();

    let actual = dot!(vlh.inverse().unwrap(), dot!(diag.r(), vlh));

    // We test the absolute distance since some imaginary parts are zero
    // making relative tests fail.

    rlst::assert_array_abs_diff_eq!(actual, a_complex, 1E-10);

    // Now check the right eigenvectors

    let vr = vr.unwrap();

    let actual = dot!(vr.r(), dot!(diag, vr.r().inverse().unwrap()));
    rlst::assert_array_abs_diff_eq!(actual, a_complex, 1E-10);
}
