//! Eigenvalues
use crate::dense::linalg::naupd::NonSymmetricArnoldiUpdate;
use crate::dense::linalg::neupd::NonSymmetricArnoldiExtract;
use crate::dense::types::RlstScalar;
use crate::operator::operations::eigenvalues::split::xslice_yslice;
use crate::operator::space::element::ElementImpl;
use crate::operator::{AsApply, Operator};
use crate::{zero_element, IndexableSpace, LinearSpace, OperatorBase};

/// Eigs parameters
pub struct Eigs<Space, OpImpl>
where
    Space: IndexableSpace,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    <Space as LinearSpace>::F: RlstScalar,
{
    operator: Operator<OpImpl>,
    howmny: String,
    bmat: String,
    ishfts: i32,
    maxiter: i32,
    mode: i32,
    tol: <Space::F as RlstScalar>::Real,
    which: String,
}

/// Order of eigenvalues
pub enum Which {
    /// Largest magnitude (default)
    LM,
    /// Smallest magnitude
    SM,
    /// Largest real part
    LR,
    /// Smallest real part
    SR,
    /// Largest imaginary part
    LI,
    /// Smallest imaginary part
    SI,
}

/// Solving mode
pub enum Mode {
    /// We are solving a standard eigenvalue problem (Ax=λx)
    Standard,
    /// We are solving a generalised eigenvalue problem Ax=λBx
    Generalised,
    /// Shift-invert mode
    ShiftInvert,
    /// Buckling transformation
    Buckling,
    /// Cayley transformation
    Cayley,
}

impl<Space, OpImpl> Eigs<Space, OpImpl>
where
    Space: IndexableSpace,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    <Space as LinearSpace>::F: RlstScalar,
    <Space as LinearSpace>::F: NonSymmetricArnoldiUpdate,
    <Space as LinearSpace>::F: NonSymmetricArnoldiExtract,
{
    /// Define configuration
    pub fn new(
        op: Operator<OpImpl>,
        tol: <Space::F as RlstScalar>::Real,
        maxiter: Option<i32>,
        mode: Option<Mode>,
        which: Option<Which>,
    ) -> Self {
        let dim = op.domain().dimension() as i32;

        let maxiter = match maxiter {
            None => 100 * dim,
            Some(maxiter) => maxiter,
        };

        let mode = match mode {
            Some(val) => match val {
                //As we implement more modes, the setup should be different
                Mode::Standard => 1,
                Mode::Generalised => panic!("Generalised mode not implemented"), // 2
                Mode::ShiftInvert => panic!("Shift-Invert mode not implemented"), // 3
                Mode::Buckling => panic!("Buckling mode not implemented"),       // 4
                Mode::Cayley => panic!("Cayley mode not implemented"),           // 5
            },
            None => 1,
        };

        let which = match which {
            Some(val) => match val {
                Which::LM => "LM".to_string(),
                Which::SM => "SM".to_string(),
                Which::LR => "LR".to_string(),
                Which::SR => "SR".to_string(),
                Which::LI => "LI".to_string(),
                Which::SI => "SI".to_string(),
            },
            None => "LM".to_string(),
        };

        Eigs {
            operator: op,
            howmny: "A".to_string(), //	All eigenvectors corresponding to the converged Ritz values are computed.
            //  'P'	A subset of eigenvectors is computed, specified via the select array.
            //  'S'	Only Schur vectors are computed (used rarely; related to internal representation).
            bmat: "I".to_string(), // Could also be B for a generalised eigenvalue problem
            ishfts: 1, // ARPACK automatically computes the shifts using exact shifts (the usual, recommended mode).
            // 0: The user provides the shifts explicitly in the shifts array.
            maxiter,
            mode,
            tol,
            which,
        }
    }

    /// Run eigs
    pub fn run(
        &mut self,
        v0: Option<&[Space::F]>,
        k: i32,
        sigma: Option<Space::F>,
        rev: bool,
    ) -> (Vec<<Space::F as RlstScalar>::Complex>, Vec<Space::F>) {
        let dim = self.operator.domain().dimension() as i32;

        assert!(k <= dim - 2, "k must be at most N-2");

        let mut ido = 0;
        let mut info = 0;
        let ncv = i32::min(i32::max(2 * k + 1, 20), dim);
        let lworkl = 3 * ncv * (ncv + 2);
        let mut resid: Vec<Space::F> = match v0 {
            None => (0..dim).map(|_| num::Zero::zero()).collect(),
            Some(v0) => (0..(dim as usize)).map(|i| v0[i]).collect(),
        };
        let mut v: Vec<Space::F> = (0..dim * ncv).map(|_| num::Zero::zero()).collect();
        let mut iparam = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        iparam[0] = self.ishfts;
        iparam[2] = self.maxiter;
        iparam[3] = 1;
        iparam[6] = self.mode;

        let mut workd: Vec<Space::F> = (0..3 * dim).map(|_| num::Zero::zero()).collect();
        let mut workl: Vec<Space::F> = (0..lworkl).map(|_| num::Zero::zero()).collect();
        let mut rwork: Vec<<Space::F as RlstScalar>::Real> =
            (0..ncv).map(|_| num::Zero::zero()).collect(); // real
        let mut ipntr: [i32; 14] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        loop {
            <Space::F as NonSymmetricArnoldiUpdate>::naupd(
                &mut ido,
                &mut self.bmat,
                dim,
                &mut self.which,
                k,
                self.tol,
                &mut resid,
                ncv,
                &mut v,
                dim,
                &mut iparam,
                &mut ipntr,
                &mut workd,
                &mut workl,
                lworkl,
                &mut rwork,
                &mut info,
            );
            let n0 = ipntr[0] - 1;
            let n1 = ipntr[1] - 1;
            let n2 = ipntr[2] - 1;

            let mut xslice_vec = zero_element(self.operator.domain());
            match ido {
                -1 => {
                    // initialization
                    let (xslice, yslice) = xslice_yslice(&mut workd, n0, n1, dim);
                    xslice_vec.imp_mut().fill_inplace_raw(xslice);
                    let mut y_slice_vec = self
                        .operator
                        .apply(xslice_vec.r(), crate::TransMode::NoTrans);
                    y_slice_vec.imp_mut().fill_raw_data(yslice);
                }
                1 => {
                    let (xslice, yslice) = xslice_yslice(&mut workd, n2, n1, dim);
                    xslice_vec.imp_mut().fill_inplace_raw(xslice);
                    let mut y_slice_vec = self
                        .operator
                        .apply(xslice_vec.r(), crate::TransMode::NoTrans);
                    y_slice_vec.imp_mut().fill_raw_data(yslice);
                }
                2 => {
                    let (xslice, yslice) = xslice_yslice(&mut workd, n0, n1, dim);
                    for i in 0..dim as usize {
                        yslice[i] = xslice[i]
                    }
                }
                3 => {
                    panic!("ARPACK requested user shifts. Assure ISHIFT==0");
                }
                _ => {
                    break;
                }
            }
        }
        if info != 0 {
            panic!("ARPACKERROR");
        }

        let mut select: Vec<i32> = (0..ncv).map(|_| 0).collect();
        let mut workev: Vec<Space::F> = (0..3 * ncv).map(|_| num::Zero::zero()).collect();
        let mut vals: Vec<<Space::F as RlstScalar>::Complex> =
            (0..k).map(|_| num::Zero::zero()).collect();
        let mut vecs: Vec<Space::F> = (0..dim * ncv).map(|_| num::Zero::zero()).collect();

        let (sigma_r, sigma_i) = match sigma {
            None => (num::Zero::zero(), num::Zero::zero()),
            Some(val) => (val.re(), val.im()),
        };

        <Space::F as NonSymmetricArnoldiExtract>::neupd(
            rev as i32,
            &self.howmny,
            &mut select,
            &mut vals,
            &mut vecs,
            dim,
            sigma_r,
            sigma_i,
            &mut workev,
            &self.bmat,
            dim,
            &self.which,
            k,
            self.tol,
            &mut resid,
            ncv,
            &mut v,
            dim,
            &mut iparam,
            &mut ipntr,
            &mut workd,
            &mut workl,
            lworkl,
            &mut rwork,
            &mut info,
        );

        if info != 0 {
            panic!("ARPACKERROR");
        }

        return (vals, vecs);
    }
}
