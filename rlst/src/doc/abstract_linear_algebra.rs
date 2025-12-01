//! # Abstract linear algebra with RLST
//!
//! RLST has a module that allows algorithms on arbitrary linear structures. The basic concept is that
//! of a function space and operators acting on function spaces. Depending on the properties of function spaces
//! different types of algorithms can be implemented.
//!
//! # Linear Spaces
//!
//! A linear space is a basic structure that can generate elements of that space and supports addition and scalar multiplication
//! operations. The associated trait is [LinearSpace](crate::traits::LinearSpace). It has two associated types, namely the
//! underlying scalar field [LinearSpace::F](crate::traits::LinearSpace::F) and the
//! implementation type [LinearSpace::Impl](crate::traits::LinearSpace::Impl) of elements of the space.
//!
//! A linear space is a very basic structure. The following traits are for spaces that provide more functionality.
//!
//! - [DualSpace](crate::traits::DualSpace): A space that has a pairing with a dual space.
//! - [IndexableSpace](crate::traits::IndexableSpace): A space with a countable finite dimensional basis.
//! - [InnerProductSpace](crate::traits::InnerProductSpace): A space that supports an inner product.
//! - [NormedSpace](crate::traits::NormedSpace): A space that supports a norm.
//!
//! Note that each `InnerProductSpace` is also a `DualSpace` and a `NormedSpace` through default implementation of the corresponding traits.
//!
//! Associated with a linear space is an [Element](crate::operator::element::Element) of the space. This concrete type is a wrapper that stores
//! an implementation type and a reference to the underlying space. The following example demonstrates how to instantiate a linear space
//! for standard vectors and perform operations on the elements.
//!
//! ```
//! use rlst::operator::interface::array_vector_space::ArrayVectorSpace;
//! use rlst::operator::space::zero_element;
//! use rlst::traits::LinearSpace;
//! # use approx::assert_relative_eq;
//! let space = ArrayVectorSpace::new(10);
//! let mut elem1 = zero_element(&space);
//! *elem1.imp_mut().get_mut([0]).unwrap() = 1.0;
//! let elem2 = space.copy_from(&elem1);
//! let elem3 = 5.0_f64 * &elem1 + &elem2;
//! # assert_relative_eq!(*elem3.imp().get([0]).unwrap(), 6.0_f64, max_relative = 1E-10);
//! ```
//! We have defined an [ArrayVectorSpace](crate::operator::interface::array_vector_space::ArrayVectorSpace) with vectors of
//! dimension `10`. We then use the [zero_element](crate::operator::space::zero_element) to create a new element in that space.
//! Using [imp_mut](crate::operator::element::Element::imp_mut) we get a mutable reference to the implementation and assign the first
//! component of the vector the value `1.0`. We then copy this element using [Space::copy](crate::traits::LinearSpace::copy_from) and
//! create a new element `elem3` as a linear combination of `elem1` and `elem2`.
//!
//! As opposed to the `Array` type in RLST, elements of vector spaces do not use lazy compile-time arithmetic. Hence, the operation `5.0 * &elem1`
//! creates a new element that is then added to `&elem2`.
//!
//! Why should we use this abstract wrapper? The advantage is that once the function space is instantiated any operation on elements of the space
//! is independent of the concrete implementation. This allows us to write algorithms that work on any linear structure and not just on specific types.
//!
//! # Operators on linear spaces
//!
//! A linear operator maps between a domain space and a range space. RLST has a complete operator arithmetic that abstracts from concrete
//! types (e.g. matrices, etc.) to mappings between linear spaces. Default implementations of operators exists for two dimensional dense arrays,
//! CSR matrices and distributed CSR matrices. To implement operators on other types one needs to implement
//! the [OperatorBase](crate::traits::abstract_operator::OperatorBase) trait. The following gives an example of using abstract operators.
//!
//! ```
//! # extern crate blas_src;
//! use rlst::DynArray;
//! use rlst::operator::abstract_operator::Operator;
//! use rlst::traits::abstract_operator::OperatorBase;
//! use rlst::operator::space::zero_element;
//! use rlst::dot;
//! # use rlst::assert_array_relative_eq;
//! let mut mat = DynArray::<f64, 2>::from_shape([20, 10]);
//! mat.fill_from_seed_equally_distributed(0);
//! let op = Operator::from(&mat);
//! let mut x = zero_element(op.domain());
//! x.imp_mut().fill_from_seed_equally_distributed(0);
//! let y = dot!(op, x);
//! # assert_array_relative_eq!(y.imp(), dot!(mat, x.imp()), 1E-10);
//! ```
//! Operators implement a complete algebra, that is we can take sums of operators and multiply them with scalars. The following gives an example.
//! ```
//! use rlst::DynArray;
//! use rlst::operator::abstract_operator::Operator;
//! use rlst::traits::abstract_operator::OperatorBase;
//! use rlst::operator::space::zero_element;
//! use rlst::dot;
//! # use rlst::assert_array_relative_eq;
//! let mut mat1 = DynArray::<f64, 2>::from_shape([20, 10]);
//! mat1.fill_from_seed_equally_distributed(0);
//!
//! let mut mat2 = DynArray::<f64, 2>::from_shape([20, 10]);
//! mat2.fill_from_seed_equally_distributed(0);
//! let op1 = Operator::from(&mat1);
//! let op2 = Operator::from(&mat2);
//!
//! let sum = op1.r() + op2.r();
//! ```
//! We take reference objects using the [Operator::r](crate::operator::abstract_operator::Operator::r) method as algebra operations on operators
//! always takes ownership.
//!
//! # Solving operator equations
//!
//! To solve abstract operator equations of the form `Ax = y`, where `A` is an operator, `y` is an object in the range space and `x` is an unnknown
//! object in the domain space, we can use iterative solvers. RLST currently only supports Conjugate Gradients with more solvers to be added.
//! The following provides a complete worked out example of how to solve operator equations with `CG`. Note that `CG` requires that the operator
//! is self-adjoint with positive eigenvalues or it may not converge.
//!
//!
//! ```
//! # extern crate blas_src;
//! # extern crate lapack_src;
//! use rand_chacha::ChaCha8Rng;
//! use rand::prelude::*;
//! use rlst::DynArray;
//! use rlst::operator::abstract_operator::Operator;
//! use rlst::operator::space::zero_element;
//! use rlst::operator::algorithms::conjugate_gradients::CgIteration;
//! use rlst::traits::abstract_operator::OperatorBase;
//! use rlst::traits::Norm;
//! let dim = 10;
//! let tol = 1E-5;
//! let mut residuals = Vec::<f64>::new();
//! let mut rng = ChaCha8Rng::seed_from_u64(0);
//! let mut mat = DynArray::<f64, 2>::from_shape([dim, dim]);
//! for index in 0..dim {
//!     mat[[index, index]] = rng.random_range(1.0..=2.0);
//! }
//! let op = Operator::from(&mat);
//! let mut rhs = zero_element(op.range());
//! rhs.imp_mut().fill_from_equally_distributed(&mut rng);
//! let mut x = zero_element(op.domain());
//! let cg = (CgIteration::new(&op, &rhs, &mut x))
//!     .set_callable(|_, res| {
//!         let res_norm = res.norm();
//!         residuals.push(res_norm);
//!     })
//!     .set_tol(tol)
//!     .print_debug();
//! let res = cg.run();
//! assert!(res < tol);
//! ```
