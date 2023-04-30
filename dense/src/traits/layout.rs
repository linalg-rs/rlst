//! Layout Definitions
//!
//! The traits in this module determine the memory layout of a matrix.
//! Consider a simple matrix of the form
//! \\[
//! A = \begin{bmatrix}1 & 2\\\\
//!                     3 & 4
//!      \end{bmatrix}
//! \\]
//!
//! In **row-major** form this matrix is stored in memory in the order
//! $\begin{bmatrix}1 & 2 & 3 & 4\end{bmatrix}$ In **column-major** form the matrix is stored in memory
//! as $\begin{bmatrix}1 & 3 & 2 & 4\end{bmatrix}$. These are the two most commonly used
//! storage formats for matrices. A more general way of describing the memory ordering of matrix
//! elements is to introduce a stride tuple `(r, c)`. The meaning is that in memory we have
//! to walk `r` positions to go from one row to the next, and `c` positions to walk from one
//! column to the next. For the matrix $A$ the stride tuple is `(2, 1)` in **row-major**
//! and `(1, 2)` in **column-major** form.
//!
//! Strides arise naturally in the description of submatrices.
//! Imagine that the matrix $A$ is a submatrix of the bigger matrix.
//! \\[
//! B = \begin{bmatrix}1 & 2 & x\\\\
//!                     3 & 4 & x\\\\
//!                     x & x & x
//!      \end{bmatrix},
//! \\]
//! where the $x$ are placeholders for some arbitrary numbers. A **row-major** layout of this matrix
//! is given as
//! \\[
//! \begin{bmatrix}1 & 2 & x & 3 & 4 & x & x & x & x\end{bmatrix}
//! \\]
//! If we wanted to describe the submatrix $A$ we could take the elements
//! $\begin{bmatrix}1 & 2 & x & 3 & 4\end{bmatrix}$. The stride tuple associated with this
//! layout is `(3, 1)`. The distance of elements within each row is `1` and the distance within
//! each column is `3`.
//!
//! However, more complicated layouts are possible (e.g. a storage layout for upper triangular
//! matrices). The trait here ensures as much generality as possible in defining memory
//! layouts.
//!
//! Each matrix is assigned with a logical indexing that is either **row-major** or **column-major**
//! and a physical layout. The logial indexing just determines whether a one-dimensional iterator
//! iterates through the matrix elements by row or by column. Consider again that $A$ is submatrix
//! of a larger $3\times 3$ matrix stored in **row-major** form. A logical **row-major** traversal
//! of $A$ will always return *\begin{bmatrix}1 & 2 & 3 & 4\end{bmatrix}$ independent of the
//! underlying physical layout.
//!
//! The logical indexing is determined by the two methods [convert_1d_2d](crate::traits::LayoutType::convert_1d_2d)
//! and [convert_2d_1d](crate::traits::LayoutType::convert_2d_1d). These methods map between
//! logical `(row, col)` index tuples and one dimensional indices. The translation to the
//! underlying physical memory locations is handled by the routines [convert_1d_raw](crate::traits::LayoutType::convert_1d_raw)
//! and [convert_2d_raw](crate::traits::LayoutType::convert_2d_raw), which convert either
//! a two dimensional `(row, col)` index or a one-dimensional index to the raw physical location.
//! The raw physical location by definition needs to have the first matrix entry as index 0.
//! For base **row-major** and **column-major** storage types the physical and logical layout
//! are typical identical. But for more complex types (e.g. with arbitrary stride vectors) they
//! are typically different from each other.
//!
//! The main trait in this module is the [LayoutType](crate::traits::LayoutType) trait.
//! If this is implemented for a matrix the [Layout](crate::traits::Layout) is auto-implemented.
//! This latter trait only provides a method to return the [LayoutType] implementation.
//! This crate also provides a number of other traits.
//!
//! The *default layout* of RLST is a column-major stride with column-major 1D logical indexing.

/// The main trait defining a layout. For detailed information see the
/// [module description](crate::traits::layout).
pub trait LayoutType {
    /// Return the stride as tuple `(r, c)` with `r` the row strice and `c` the column stride.
    fn stride(&self) -> (usize, usize);

    /// Return the dimension of the matrix as tuple `(rows, cols)`.
    fn dim(&self) -> (usize, usize);

    /// The number of elements in the matrix or vector.
    fn number_of_elements(&self) -> usize;

    /// Convert a 1d logical `index` to a 2d `(row, col)` index.
    fn convert_1d_2d(&self, index: usize) -> (usize, usize);

    /// Convert a 2d logical `(row, col)` index to a 1d logical `index`.
    fn convert_2d_1d(&self, row: usize, col: usize) -> usize;

    /// Convert a 1d logical `index` to a raw memory index.
    fn convert_1d_raw(&self, index: usize) -> usize;

    /// Convert a 2d logical `(row, col)` index to a raw memory index.
    fn convert_2d_raw(&self, row: usize, col: usize) -> usize;

    /// Create a new layout from providing the matrix dimension as `(rows, cols)` tuple.
    fn from_dimension(dim: (usize, usize), stride: (usize, usize)) -> Self;
}

/// This layout provides a method to return layout information.
/// It is auto-implemented for all objects that implement [LayoutType].
pub trait Layout {
    type Impl: LayoutType;

    /// Return the associated layout.
    fn layout(&self) -> &Self::Impl;
}
