//! Implementation of matrix conjugate

use crate::op_containers::complex_mat::{ComplexContainer, ComplexMat};
use crate::MatrixImplTrait;
use crate::{Matrix, SizeIdentifier};
use rlst_common::traits::ToComplex;

pub use rlst_common::types::{c32, c64, Scalar};

impl<MatImpl: MatrixImplTrait<f64, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier> ToComplex
    for Matrix<f64, MatImpl, RS, CS>
{
    type Out = ComplexMat<c64, MatImpl, RS, CS>;

    fn to_complex(self) -> Self::Out {
        Matrix::new(ComplexContainer::<c64, MatImpl, RS, CS>::new(self))
    }
}

impl<MatImpl: MatrixImplTrait<f32, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier> ToComplex
    for Matrix<f32, MatImpl, RS, CS>
{
    type Out = ComplexMat<c32, MatImpl, RS, CS>;

    fn to_complex(self) -> Self::Out {
        Matrix::new(ComplexContainer::<c32, MatImpl, RS, CS>::new(self))
    }
}

impl<MatImpl: MatrixImplTrait<c32, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier> ToComplex
    for Matrix<c32, MatImpl, RS, CS>
{
    type Out = Self;

    fn to_complex(self) -> Self::Out {
        self
    }
}

impl<MatImpl: MatrixImplTrait<c64, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier> ToComplex
    for Matrix<c64, MatImpl, RS, CS>
{
    type Out = Self;

    fn to_complex(self) -> Self::Out {
        self
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::rlst_rand_mat;
    use rlst_common::traits::*;
    use rlst_common::types::c64;

    #[test]
    fn test_to_complex_f64() {
        let mat = rlst_rand_mat![f64, (3, 4)];

        let complex_mat = mat.view().to_complex();

        for (complex, real) in complex_mat.iter_col_major().zip(mat.iter_col_major()) {
            assert_eq!(complex, c64::from_real(real));
        }
    }

    #[test]
    fn test_to_complex_f32() {
        let mat = rlst_rand_mat![f32, (3, 4)];

        let complex_mat = mat.view().to_complex();

        for (complex, real) in complex_mat.iter_col_major().zip(mat.iter_col_major()) {
            assert_eq!(complex, c32::from_real(real));
        }
    }

    #[test]
    fn test_to_complex_c32() {
        let mat = rlst_rand_mat![c32, (3, 4)];

        let complex_mat = mat.view().to_complex();

        for (complex, complex_orig) in complex_mat.iter_col_major().zip(mat.iter_col_major()) {
            assert_eq!(complex, complex_orig);
        }
    }

    #[test]
    fn test_to_complex_c64() {
        let mat = rlst_rand_mat![c64, (3, 4)];

        let complex_mat = mat.view().to_complex();

        for (complex, complex_orig) in complex_mat.iter_col_major().zip(mat.iter_col_major()) {
            assert_eq!(complex, complex_orig);
        }
    }
}
