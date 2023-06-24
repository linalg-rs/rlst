//! Interface to Blis types
use crate::raw;

#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum TransMode {
    ConjNoTrans = raw::trans_t_BLIS_CONJ_NO_TRANSPOSE,
    NoTrans = raw::trans_t_BLIS_NO_TRANSPOSE,
    Trans = raw::trans_t_BLIS_TRANSPOSE,
    ConjTrans = raw::trans_t_BLIS_CONJ_TRANSPOSE,
}

// pub trait BlisIdentifier {
//     const ID: u32;
// }

// impl BlisIdentifier for f32 {
//     const ID: u32 = raw::num_t_BLIS_FLOAT;
// }

// impl BlisIdentifier for f64 {
//     const ID: u32 = raw::num_t_BLIS_DOUBLE;
// }

// impl BlisIdentifier for c32 {
//     const ID: u32 = raw::num_t_BLIS_SCOMPLEX;
// }

// impl BlisIdentifier for c64 {
//     const ID: u32 = raw::num_t_BLIS_DCOMPLEX;
// }

// pub struct BlisObject {
//     obj: raw::obj_t,
//     requires_free: bool,
// }

// impl Drop for BlisObject {
//     fn drop(&mut self) {
//         if self.requires_free {
//             unsafe {
//                 crate::raw::bli_obj_free(&mut self.obj);
//             }
//         }
//     }
// }

// impl Default for raw::obj_t {
//     fn default() -> Self {
//         Self {
//             root: std::ptr::null_mut(),
//             off: [0, 0],
//             dim: [0, 0],
//             diag_off: 0,
//             info: 0,
//             info2: 0,
//             elem_size: 0,
//             buffer: std::ptr::null_mut(),
//             rs: 0,
//             cs: 0,
//             is: 0,
//             scalar: raw::dcomplex {
//                 real: 0.0,
//                 imag: 0.0,
//             },
//             m_padded: 0,
//             n_padded: 0,
//             ps: 0,
//             pd: 0,
//             m_panel: 0,
//             n_panel: 0,
//             pack_fn: None,
//             pack_params: std::ptr::null_mut(),
//             ker_fn: None,
//             ker_params: std::ptr::null_mut(),
//         }
//     }
// }

// impl BlisObject {
//     pub fn from_slice<T: Scalar + BlisIdentifier>(
//         data: &mut [T],
//         stride: (usize, usize),
//         shape: (usize, usize),
//     ) -> Self {
//         // The maximum index that still needs to fit in the data slice.
//         let max_index = stride.0 * (shape.0 - 1) + stride.1 * (shape.1 - 1);
//         assert_eq!(
//             data.len(),
//             1 + max_index,
//             "Length of slice is {} but expected {}",
//             data.len(),
//             1 + max_index
//         );

//         let mut obj = raw::obj_t::default();

//         unsafe {
//             raw::bli_obj_create_with_attached_buffer(
//                 T::ID,
//                 shape.0 as i64,
//                 shape.1 as i64,
//                 data.as_mut_ptr() as *mut std::ffi::c_void,
//                 stride.0 as i64,
//                 stride.1 as i64,
//                 &mut obj,
//             )

//         };

//         BlisObject {
//             obj,
//             requires_free: false,
//         }
//     }

//     pub fn from_scalar<T: Scalar + BlisIdentifier>(scalar: T) -> Self {
//         let mut obj = raw::obj_t::default();
//         unsafe { raw::bli_obj_create_1x1(T::ID, &mut obj) };

//         unsafe {
//             raw::bli_setsc(
//                 num::cast::<T::Real, f64>(scalar.re()).unwrap(),
//                 num::cast::<T::Real, f64>(scalar.im()).unwrap(),
//                 &obj,
//             )
//         };

//         Self {
//             obj,
//             requires_free: true,
//         }
//     }

//     pub fn get_obj(&self) -> &raw::obj_t {
//         &self.obj
//     }
// }
