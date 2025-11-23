//! Implementation of the QR decomposition.

use crate::UnsafeRandom1DAccessByValue;
use crate::base_types::RlstResult;
use crate::dense::array::{Array, DynArray};
use crate::traits::base_operations::Shape;
use crate::traits::linalg::decompositions::Qr;
use crate::traits::linalg::lapack::Lapack;

use super::interface::geqp3::Geqp3;
use super::interface::geqrf::Geqrf;
use super::interface::orgqr::Orgqr;

/// Stores the result of a QR decomposition of a matrix.
pub struct QrDecomposition<Item> {
    a: DynArray<Item, 2>,
    jpvt: Vec<i32>,
    tau: Vec<Item>,
}

/// The pivoting strategy for the QR decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnablePivoting {
    /// Column pivoting,
    Yes,
    /// No pivoting.
    No,
}

impl<Item, ArrayImpl> Qr for Array<ArrayImpl, 2>
where
    Item: Lapack,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
{
    type Item = Item;

    /// Compute the QR decomposition of a matrix.
    fn qr(&self, pivoting: EnablePivoting) -> RlstResult<QrDecomposition<Item>> {
        let mut a = DynArray::new_from(self);
        let (m, n, lda) = (a.shape()[0], a.shape()[1], a.shape()[0]);
        let k = std::cmp::min(m, n);

        let mut jpvt = vec![0_i32; n];
        let mut tau = vec![Item::zero(); k];

        match pivoting {
            EnablePivoting::No => {
                for (i, item) in jpvt.iter_mut().enumerate() {
                    *item = 1 + i as i32;
                }
                <Item as Geqrf>::geqrf(m, n, a.data_mut().unwrap(), lda, &mut tau)?;
                Ok(QrDecomposition { a, jpvt, tau })
            }
            EnablePivoting::Yes => {
                <Item as Geqp3>::geqp3(m, n, a.data_mut().unwrap(), lda, &mut jpvt, &mut tau)?;
                Ok(QrDecomposition { a, jpvt, tau })
            }
        }
    }
}

/// The mode for computing the Q matrix in a QR decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QMode {
    /// Compute the full Q matrix.
    Full,
    /// Compute the compact Q matrix.
    Compact,
}

impl<Item> QrDecomposition<Item>
where
    Item: Lapack,
{
    /// Compute the Q matrix from the QR decomposition.
    /// /// The `mode` parameter determines whether to compute the full or compact Q matrix.
    /// - `QMode::Full`: Computes the full Q matrix with shape (m, m).
    /// - `QMode::Compact`: Computes the compact Q matrix with shape (m, min(m, n)).
    pub fn q_mat(&self, mode: QMode) -> RlstResult<DynArray<Item, 2>> {
        let [m, n] = self.a.shape();
        let k = std::cmp::min(m, n);
        let (m, n) = match mode {
            QMode::Full => (m, m), // Full Q matrix has m rows and m columns
            QMode::Compact => (m, k),
        };

        let mut q = DynArray::<Item, 2>::from_shape([m, n]);

        // Copy the elementary reflectors from `self.a` to `q`
        for col in 0..k {
            for row in 1 + col..m {
                unsafe {
                    *q.get_unchecked_mut([row, col]) = *self.a.get_unchecked([row, col]);
                }
            }
        }

        <Item as Orgqr>::orgqr(m, n, k, q.data_mut().unwrap(), m, &self.tau)?;

        Ok(q)
    }

    /// Return the R matrix of the QR decomposition.
    /// The R matrix is of dimension (min(m, n), n).
    pub fn r_mat(&self) -> RlstResult<DynArray<Item, 2>> {
        let [m, n] = self.a.shape();
        let k = std::cmp::min(m, n);

        let mut r = DynArray::<Item, 2>::from_shape([k, n]);

        for col in 0..n {
            for row in 0..std::cmp::min(1 + col, m) {
                unsafe {
                    *r.get_unchecked_mut([row, col]) = *self.a.get_unchecked([row, col]);
                }
            }
        }

        Ok(r)
    }

    /// Return the permutation matrix P such at `A * P = Q * R`.
    pub fn p_mat(&self) -> RlstResult<DynArray<Item, 2>> {
        let [_, n] = self.a.shape();
        let mut p_mat = DynArray::<Item, 2>::from_shape([n, n]);

        for (i, &j) in self.jpvt.iter().enumerate() {
            unsafe {
                *p_mat.get_unchecked_mut([j as usize - 1, i]) = Item::one();
            }
        }

        Ok(p_mat)
    }

    /// Return the pivot indices of the QR decomposition.
    /// If `jpvt[i] = j`, then the `i`-th column of the QR decomposition corresponds to the `j`-th
    /// column of the original matrix.
    /// The indices are returned in zero-based indexing.
    pub fn perm(&self) -> Vec<usize> {
        self.jpvt
            .iter()
            .map(|&i| i as usize - 1) // Convert to zero-based indexing
            .collect()
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::MultIntoResize;
    use crate::base_types::TransMode;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::empty_array;
    use paste::paste;

    macro_rules! implement_qr_tests {
    ($scalar:ty, $tol:expr) => {
        paste! {

        #[test]
        pub fn [<test_thin_qr_$scalar>]() {
            let mut mat = DynArray::<$scalar, 2>::from_shape([8, 5]);

            mat.fill_from_seed_equally_distributed(0);

            let mut ident = DynArray::<$scalar, 2>::from_shape([5, 5]);
            ident.set_identity();

            let qr = mat.qr(EnablePivoting::No).unwrap();

            let q_mat = qr.q_mat(QMode::Compact).unwrap();
            let r_mat = qr.r_mat().unwrap();
            let p_t_mat = DynArray::new_from(&qr.p_mat().unwrap().transpose());

            let actual = crate::empty_array().simple_mult_into_resize(empty_array().simple_mult_into_resize(q_mat.r(), r_mat.r()), p_t_mat.r());

            crate::assert_array_relative_eq!(actual, mat, $tol);

            let qtq = crate::empty_array().mult_into_resize(
                TransMode::ConjTrans,
                TransMode::NoTrans,
                1.0.into(),
                q_mat.r(),
                q_mat.r(),
                1.0.into(),
            );

            crate::assert_array_abs_diff_eq!(qtq, ident, $tol);
        }


        #[test]
        pub fn [<test_thick_qr_$scalar>]() {
            let mut mat = DynArray::<$scalar, 2>::from_shape([5, 8]);

            mat.fill_from_seed_equally_distributed(0);

            let mut ident = DynArray::<$scalar, 2>::from_shape([5, 5]);
            ident.set_identity();

            let qr = mat.qr(EnablePivoting::No).unwrap();

            let q_mat = qr.q_mat(QMode::Compact).unwrap();
            let r_mat = qr.r_mat().unwrap();
            let p_t_mat = DynArray::new_from(&qr.p_mat().unwrap().transpose());

            let actual = empty_array()
                .simple_mult_into_resize(empty_array().simple_mult_into_resize(q_mat.r(), r_mat.r()), p_t_mat.r());

            crate::assert_array_relative_eq!(actual, mat, $tol);

            let qtq = empty_array().mult_into_resize(
                TransMode::ConjTrans,
                TransMode::NoTrans,
                1.0.into(),
                q_mat.r(),
                q_mat.r(),
                1.0.into(),
            );

            crate::assert_array_abs_diff_eq!(qtq, ident, $tol);
        }


        #[test]
        pub fn [<test_thin_qr_pivoted_$scalar>]() {
            let mut mat = DynArray::<$scalar, 2>::from_shape([8, 5]);

            mat.fill_from_seed_equally_distributed(0);

            let mut ident = DynArray::<$scalar, 2>::from_shape([5, 5]);
            ident.set_identity();

            let qr = mat.qr(EnablePivoting::Yes).unwrap();

            let q_mat = qr.q_mat(QMode::Compact).unwrap();
            let r_mat = qr.r_mat().unwrap();
            let p_t_mat = DynArray::new_from(&qr.p_mat().unwrap().transpose());

            let actual = empty_array()
                .simple_mult_into_resize(empty_array().simple_mult_into_resize(q_mat.r(), r_mat.r()), p_t_mat.r());

            crate::assert_array_relative_eq!(actual, mat, $tol);

            let qtq = empty_array().mult_into_resize(
                TransMode::ConjTrans,
                TransMode::NoTrans,
                1.0.into(),
                q_mat.r(),
                q_mat.r(),
                1.0.into(),
            );

            crate::assert_array_abs_diff_eq!(qtq, ident, $tol);
        }


        #[test]
        pub fn [<test_thick_qr_pivoted_$scalar>]() {
            let mut mat = DynArray::<$scalar, 2>::from_shape([5, 8]);

            mat.fill_from_seed_equally_distributed(0);

            let mut ident = DynArray::<$scalar, 2>::from_shape([5, 5]);
            ident.set_identity();

            let qr = mat.qr(EnablePivoting::Yes).unwrap();

            let q_mat = qr.q_mat(QMode::Compact).unwrap();
            let r_mat = qr.r_mat().unwrap();
            let p_t_mat = DynArray::new_from(&qr.p_mat().unwrap().transpose());

            let actual = empty_array()
                .simple_mult_into_resize(empty_array().simple_mult_into_resize(q_mat.r(), r_mat.r()), p_t_mat.r());

            crate::assert_array_relative_eq!(actual, mat, $tol);

            let qtq = empty_array().mult_into_resize(
                TransMode::ConjTrans,
                TransMode::NoTrans,
                1.0.into(),
                q_mat.r(),
                q_mat.r(),
                1.0.into(),
            );

            crate::assert_array_abs_diff_eq!(qtq, ident, $tol);
        }




                }
    };
}

    implement_qr_tests!(f32, 1E-5);
    implement_qr_tests!(f64, 1E-10);
    implement_qr_tests!(c32, 1E-4);
    implement_qr_tests!(c64, 1E-10);
}
