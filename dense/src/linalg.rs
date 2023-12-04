//! Linear algebra routines
pub mod inverse;
pub mod lu;
pub mod pseudo_inverse;
pub mod qr;
pub mod svd;

pub fn assert_lapack_stride(stride: [usize; 2]) {
    assert_eq!(
        stride[0], 1,
        "Incorrect stride for Lapack. Stride[0] is {} but expected 1.",
        stride[0]
    );
}

pub enum Trans {
    Trans,
    NoTrans,
    ConjTrans,
}
