//!  Examples for using sparse matrices.

use rlst::{self, sparse::csr_mat::CsrMatrix, AijIteratorByValue, ArrayOpAbs, FromAij, Shape};

fn main() {
    // This is a placeholder for the main function.
    // You can add tests or examples of how to use the traits defined above.
    let mut arr = rlst::rlst_dynamic_array!(f64, [4, 5]);
    arr.fill_from_seed_equally_distributed(0);

    let rows = vec![0, 1, 2];
    let cols = vec![0, 2, 1];
    let data = vec![-1.0, 2.0, 3.0];

    let csr_mat = CsrMatrix::from_aij([3, 3], &rows, &cols, &data);

    println!("CSR Matrix Shape: {:#?}", csr_mat.shape());

    for ([i, j], v) in csr_mat.iter_aij_value() {
        println!("Row: {}, Col: {}, Value: {}", i, j, v);
    }

    println!("Apply operator.");
    let csr_abs = csr_mat.op().abs();
}
