//! Operator Algebra
//!
//! RLST supports a generic operator algebra based on the concept of function spaces and linear
//! operators acting on them. This allows the expression of operations on arbitrary linear spaces,
//! e.g. polynomials, matrices, etc. through RLST, and the application of generic iterative solvers
//! acting on them.
//!
//! # Function spaces
//!
//! The most basic concept is that of a function space. The trait [LinearSpace](crate::LinearSpace)
//! defines the concept of a linear function space, consisting of a, field type
//! [LinearSpace::F](crate::LinearSpace::F), an element type
//! [LinearSpace::E](crate::LinearSpace::E) and simple factory routines to produce a zero element
//! ([zero](crate::LinearSpace::zero)) and to create a new element as copy from an existing
//! element ([new_from](crate::LinearSpace::new_from)).
//!
//! In most applications one deals with normed or inner product spaces. We define the following
//! additional traits that depend on [LinearSpace](crate::LinearSpace).
//!
//! - [NormedSpace](crate::NormedSpace): A function space in which elements have a norm.
//! - [DualSpace](crate::DualSpace): A function space defining a dual pairing with another space.
//! - [InnerProductSpace](crate::InnerProductSpace): A function space with an inner product. An
//! inner product space also automatically implements the [NormedSpace](crate::NormedSpace) and
//! [DualSpace](crate::DualSpace) traits,
//! where the norm is defined through the squareroot of the inner product of an element with
//! itself and the dual pairing is defined as the inner product with itself.
//!
//! # Elements of function spaces
//!
//! Elements of function spaces support the usual linear operations. These are
//! - [axpy_inplace](crate::Element::axpy_inplace): y -> alpha * x + y
//! - [sum_inplace](crate::Element::sum_inplace): y -> x + y
//! - [fill_inplace](crate::Element::fill_inplace): x -> y
//! - [scale_inplace](crate::Element::scale_inplace): y -> alpha * y
//! - [neg_inplace](crate::Element::neg_inplace): y -> -y
//!
//! For convenience there are also corresponding methods provided without the `_inplace` ending,
//! which take ownership of `y` and return the modified `y`. Default implementations of these
//! routines internally use the `_inplace` versions and then return the objects again.
//!
//! To **view** and **modify** the contents of an element the [view](crate::Element::view) and
//! [view_mut](crate::Element::view_mut) methods are provided. These return implementation
//! dependent views onto the data of an element of the space.
//!
//! # Operators
//!
//! Traits are provided for operators acting on linear spaces. An operator `A` is a linear map
//! that maps from a space `X` to a space `Y`, Here, `X` is the `domain` and `Y` is the `range`.
//! The [OperatorBase](crate::OperatorBase) trait defines the basic properties of operators, in
//! particular it defines the following methods.
//! - [domain](crate::OperatorBase::domain): Return a reference of the domain space.
//! - [range](crate::OperatorBase::range): Return a reference of the range space.
//! - [scale](crate::OperatorBase::scale): Scale the operator.
//! - [sum](crate::OperatorBase::sum): Sum with another operator.
//! - [product](crate::OperatorBase::product): Form the product with another operator.
//!
//! The `OperatorBase` trait does not allow application to an element. Operators that can be applied
//! to elements of the space implement the `AsApply` trait, which derives from `OperatorBase`.
//!
//! # Iterative solvers for abstract operators
//!
//! The concepts of a linear space, elements of the space, and operators acting on the space is
//! sufficient to implement iterative solvers. The advantage of this abstract concept is that we
//! can use the same iterative solver, no matter whether the underlying objects are standard
//! vector spaces with dense matrices, sparse matrix algebras, or more complex objects such as
//! function spaces.
//!
//! Currently, only Conjugate Gradients is implemented. The following example demonstrates CG
//! applied to a dense matrix acting on a vector space.
//! ```
//! # extern crate blas_src;
//! # extern crate lapack_src;
//! # use rlst::prelude::*;
//! # use rand;
//! # use rand::prelude::*;
//! let dim = 10;
//! let tol = 1E-5;
//!
//! let space = ArrayVectorSpace::<f64>::new(dim);
//! let mut residuals = Vec::<f64>::new();
//! let mut rng = rand::thread_rng();
//!
//! let mut mat = rlst_dynamic_array2!(f64, [dim, dim]);
//!
//! for index in 0..dim {
//!     mat[[index, index]] = rng.gen_range(1.0..=2.0);
//! }
//!
//! let op = DenseMatrixOperator::new(mat.view(), &space, &space);
//!
//! let mut rhs = space.zero();
//! rhs.view_mut().fill_from_equally_distributed(&mut rng);
//!
//! let cg = (CgIteration::new(&op, &rhs))
//!     .set_callable(|_, res| {
//!         let res_norm = space.norm(res);
//!         residuals.push(res_norm);
//!     })
//!     .set_tol(tol)
//!     .print_debug();
//!
//! let (_sol, res) = cg.run();
//! assert!(res < tol);
//! ```
