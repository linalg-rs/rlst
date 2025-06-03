//! Lapack LU Decomposition.

use lapack::dgetrf;

use crate::dense::types::{RlstError, RlstResult};

use super::LapackWrapperMut;

/// A trait for computing the LU decomposition of a matrix in place.
pub trait LapackLu<'a>
where
    for<'b> LuDecomposition<'b, Self::Item>: ComputedLu,
{
    /// The item type contained in the matrix.
    type Item;

    /// Compute the LU Decomposition of the matrix.
    fn lu(self) -> RlstResult<LuDecomposition<'a, Self::Item>>;
}

/// Trait for functions on a computed LU decomposition.
pub trait ComputedLu {
    /// The item type of the LU decomposition.
    type Item;
    /// Return the LU decomposition data.
    fn lu_data(&self) -> &LapackWrapperMut<'_, Self::Item>;

    /// Return the pivot indices. Indices are 1-based.
    fn ipiv(&self) -> &[i32];

    /// Compute the determinant of the matrix.
    fn det(&self) -> Self::Item;
}

/// Store the result of an LU decomposition of a matrix.
pub struct LuDecomposition<'a, Item> {
    lu: LapackWrapperMut<'a, Item>,
    ipiv: Vec<i32>,
}

impl<'a> LapackLu<'a> for LapackWrapperMut<'a, f64> {
    type Item = f64;

    fn lu(self) -> RlstResult<LuDecomposition<'a, f64>> {
        let (m, n, lda) = (self.m, self.n, self.lda);

        let dim = std::cmp::min(m, n);
        if dim == 0 {
            return Err(RlstError::MatrixIsEmpty((m as usize, n as usize)));
        }
        let mut ipiv = vec![0; dim as usize];
        let mut info = 0;
        unsafe {
            dgetrf(m, n, self.data, lda, &mut ipiv, &mut info);
        }

        match info {
            0 => Ok(LuDecomposition { lu: self, ipiv }),
            _ => Err(RlstError::LapackError(info)),
        }
    }
}

impl<'a> ComputedLu for LuDecomposition<'a, f64> {
    type Item = f64;

    fn lu_data(&self) -> &LapackWrapperMut<'_, Self::Item> {
        &self.lu
    }

    fn ipiv(&self) -> &[i32] {
        &self.ipiv
    }

    fn det(&self) -> Self::Item {
        assert_eq!(
            self.lu.m, self.lu.n,
            "Matrix must be square to compute determinant."
        );
        let mut det = self.lu.data[0];
        if self.ipiv[0] != 1 {
            det = -det;
        }
        for i in 1..self.lu.m as usize {
            det *= self.lu.data[i * self.lu.lda as usize + i];
            if self.ipiv[i] != (i + 1) as i32 {
                det = -det;
            }
        }
        det
    }
}
