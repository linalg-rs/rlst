//! Test the matrix market format

use rlst_common::traits::{AijIterator, ColumnMajorIterator, RawAccess, Shape};
use rlst_dense::rlst_rand_mat;
use rlst_io::matrix_market::{
    read_array_mm, read_coordinate_mm, write_array_mm, write_coordinate_mm,
};
use tempfile::NamedTempFile;

#[test]
fn read_mm_coo_file() {
    let pathname = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/", "test_mat_53_27.mtx");
    let mat = read_coordinate_mm::<f64>(pathname).unwrap();

    assert_eq!(mat.shape(), (53, 27));
    assert_eq!(mat.data().len(), 143);
}

#[test]
fn read_mm_array_file() {
    let pathname = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/",
        "test_array_70_30.mtx"
    );
    let mat = read_array_mm::<f64>(pathname).unwrap();

    assert_eq!(mat.shape(), (70, 30));
    assert_eq!(mat.data().len(), 70 * 30);
}

#[test]
fn write_mm_array_file() {
    // The following creates a proper temporary file to write into.
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.into_temp_path();
    let pathname = path.to_str().unwrap();

    let mat = rlst_rand_mat!(rlst_common::types::c64, (5, 4));
    write_array_mm(&mat, pathname).unwrap();

    let mat_in = read_array_mm::<rlst_common::types::c64>(pathname).unwrap();

    for (expected, actual) in mat.iter_col_major().zip(mat_in.iter_col_major()) {
        assert_eq!(expected, actual);
    }

    path.close().unwrap();
}

#[test]
fn write_mm_coordinate_file() {
    // The following creates a proper temporary file to write into.
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.into_temp_path();
    let pathname = path.to_str().unwrap();

    let rows = vec![2, 3, 4, 4, 6];
    let cols = vec![0, 1, 0, 2, 1];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // This matrix has a zero row at the beginning one in between and several zero rows at the end.
    let csr =
        rlst_sparse::sparse::csr_mat::CsrMatrix::from_aij((10, 3 ), &rows, &cols, &data).unwrap();

    write_coordinate_mm(&csr, pathname).unwrap();

    let csr_in = read_coordinate_mm::<f64>(pathname).unwrap();

    for (expected, actual) in csr.iter_aij().zip(csr_in.iter_aij()) {
        assert_eq!(expected, actual);
    }

    path.close().unwrap();
}
