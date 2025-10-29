//! In this example we demonstrate how to use the LU decomposition.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rlst::{DynArray, Lu};

fn main() {
    // The dimension of the matrix.
    let n = 10;

    // Initalize the random number generator.

    let mut rng = ChaCha8Rng::seed_from_u64(0);

    // Create a new n x n array.
    let mut arr = DynArray::<f64, 2>::from_shape([n, n]);

    // Fill the array with normally distributed random values.
    arr.fill_from_standard_normal(&mut rng);

    // We can now compute the LU decomposition.

    let lu = arr.lu().unwrap();

    // We can use the LU to solve a linear system of equations.

    let mut rhs = DynArray::<f64, 1>::from_shape([n]);
    rhs.fill_from_standard_normal(&mut rng);

    let _x = lu.solve(rlst::TransMode::NoTrans, &rhs).unwrap();

    // We can also compute the determinant.

    let _det = lu.det();
}
