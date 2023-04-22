//! Test the matrix market format

use rlst_io::matrix_market::{read_array_mm, read_coordinate_mm};

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

    assert_eq!(mat.dim(), (70, 30));
    assert_eq!(mat.data().len(), 70 * 30);
}
