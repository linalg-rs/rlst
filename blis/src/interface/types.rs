// //! Interface to Blis types
// use crate::raw;

// /// Transposition Mode.
// #[derive(Clone, Copy, PartialEq)]
// #[repr(u32)]
// pub enum TransMode {
//     /// Complex conjugate of matrix.
//     ConjNoTrans = raw::trans_t_BLIS_CONJ_NO_TRANSPOSE,
//     /// No modification of matrix.
//     NoTrans = raw::trans_t_BLIS_NO_TRANSPOSE,
//     /// Transposition of matrix.
//     Trans = raw::trans_t_BLIS_TRANSPOSE,
//     /// Conjugate transpose of matrix.
//     ConjTrans = raw::trans_t_BLIS_CONJ_TRANSPOSE,
// }
