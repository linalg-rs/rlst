// //! Fixed size matrix example
// //!

// use rlst_dense::Shape;
// use rlst_dense::SizeIdentifier;
// use rlst_proc_macro::rlst_static_size;

// #[rlst_static_size(33, 4)]
// pub struct MySizeType;

// pub fn main() {
//     match MySizeType::SIZE {
//         rlst_dense::SizeIdentifierValue::Static(m, n) => println!("{:#?}", (m, n)),
//         rlst_dense::SizeIdentifierValue::Dynamic => println!("Dynamic"),
//     }

//     let mat = rlst_dense::rlst_static_mat!(f64, MySizeType);
//     let mat_dynamic: rlst_dense::Matrix<f64, _, _> = rlst_dense::rlst_dynamic_mat!(f64, (30, 40));

//     println!("{:#?}", mat.shape());
//     println!("{:#?}", mat_dynamic.shape());
// }

pub fn main() {}
