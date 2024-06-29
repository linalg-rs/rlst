//! Operations on arrays.
use crate::dense::linalg;
use crate::{dense::types::RlstResult, TransMode};
use num::Zero;

use crate::dense::{layout::convert_1d_nd_from_shape, traits::MatrixSvd};

use super::{
    Array, ChunkedAccess, DefaultIterator, DefaultIteratorMut, RandomAccessByValue,
    RandomAccessMut, RawAccessMut, RlstScalar, Shape, Stride, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::dense::traits::ResizeInPlace;

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Set all elements of an array to zero.
    pub fn set_zero(&mut self) {
        for elem in self.iter_mut() {
            *elem = <Item as num::Zero>::zero();
        }
    }

    /// Set all elements of an array to one.
    pub fn set_one(&mut self) {
        for elem in self.iter_mut() {
            *elem = <Item as num::One>::one();
        }
    }

    /// Fill the diagonal of an array with the value 1 and all other elements zero.
    pub fn set_identity(&mut self) {
        self.set_zero();

        for index in 0..self.shape().iter().copied().min().unwrap() {
            *self.get_mut([index; NDIM]).unwrap() = <Item as num::One>::one();
        }
    }

    /// Multiply all array elements with the scalar `alpha`.
    pub fn scale_inplace(&mut self, alpha: Item) {
        for elem in self.iter_mut() {
            *elem *= alpha;
        }
    }

    /// Get the diagonal of an array.
    ///
    /// Argument must be a 1d array of length `self.shape().iter().min()`.
    pub fn get_diag<
        ArrayImplOther: UnsafeRandomAccessByValue<1, Item = Item>
            + Shape<1>
            + UnsafeRandomAccessMut<1, Item = Item>,
    >(
        &self,
        mut other: Array<Item, ArrayImplOther, 1>,
    ) {
        assert_eq!(
            other.number_of_elements(),
            *self.shape().iter().min().unwrap()
        );
        for index in 0..self.shape().iter().copied().min().unwrap() {
            *other.get_mut([index]).unwrap() = self.get_value([index; NDIM]).unwrap();
        }
    }

    /// Set the diagonal of an array.
    ///
    /// Argument must be a 1d array of length `self.shape().iter().min()`.
    pub fn set_diag<
        ArrayImplOther: UnsafeRandomAccessByValue<1, Item = Item>
            + Shape<1>
            + UnsafeRandomAccessMut<1, Item = Item>,
    >(
        &mut self,
        other: Array<Item, ArrayImplOther, 1>,
    ) {
        assert_eq!(
            other.number_of_elements(),
            *self.shape().iter().min().unwrap()
        );
        for index in 0..self.shape().iter().copied().min().unwrap() {
            *self.get_mut([index; NDIM]).unwrap() = other.get_value([index]).unwrap();
        }
    }

    /// Fill an array with values from another array.
    pub fn fill_from<ArrayImplOther: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>>(
        &mut self,
        other: Array<Item, ArrayImplOther, NDIM>,
    ) {
        assert_eq!(self.shape(), other.shape());

        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item = other_item;
        }
    }

    /// Fill an array from another array and resize if necessary.
    pub fn fill_from_resize<
        ArrayImplOther: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    >(
        &mut self,
        other: Array<Item, ArrayImplOther, NDIM>,
    ) where
        Self: ResizeInPlace<NDIM>,
    {
        if self.shape() != other.shape() {
            self.resize_in_place(other.shape());
        }

        self.fill_from(other)
    }

    /// Fill an array with values from an other arrays using chunks of size `N`.
    pub fn fill_from_chunked<
        Other: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const N: usize,
    >(
        &mut self,
        other: Other,
    ) {
        assert_eq!(self.shape(), other.shape());

        let mut chunk_index = 0;

        while let Some(chunk) = other.get_chunk(chunk_index) {
            let data_start = chunk.start_index;

            for data_index in 0..chunk.valid_entries {
                unsafe {
                    *self.get_unchecked_mut(convert_1d_nd_from_shape(
                        data_start + data_index,
                        self.shape(),
                    )) = chunk.data[data_index];
                }
            }
            chunk_index += 1;
        }
    }

    /// Fill an array with values from an other arrays using chunks of size `N`. Resize if necessary.
    pub fn fill_from_chunked_resize<
        Other: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const N: usize,
    >(
        &mut self,
        other: Other,
    ) where
        Self: ResizeInPlace<NDIM>,
    {
        if self.shape() != other.shape() {
            self.resize_in_place(other.shape());
        }

        self.fill_from_chunked::<_, N>(other)
    }

    /// Sum other array into array.
    pub fn sum_into<ArrayImplOther: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>>(
        &mut self,
        other: Array<Item, ArrayImplOther, NDIM>,
    ) {
        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item += other_item;
        }
    }

    /// Componentwise multiply other array into array.
    pub fn cmp_mult_into<
        ArrayImplOther: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    >(
        &mut self,
        other: Array<Item, ArrayImplOther, NDIM>,
    ) {
        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item *= other_item;
        }
    }

    /// Chunked summation into array.
    pub fn sum_into_chunked<
        ArrayImplOther: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const N: usize,
    >(
        &mut self,
        other: Array<Item, ArrayImplOther, NDIM>,
    ) where
        Self: ChunkedAccess<N, Item = Item>,
    {
        assert_eq!(self.shape(), other.shape());

        let mut chunk_index = 0;

        while let (Some(mut my_chunk), Some(chunk)) =
            (self.get_chunk(chunk_index), other.get_chunk(chunk_index))
        {
            let data_start = chunk.start_index;

            for data_index in 0..chunk.valid_entries {
                my_chunk.data[data_index] += chunk.data[data_index];
            }

            for data_index in 0..chunk.valid_entries {
                unsafe {
                    *self.get_unchecked_mut(convert_1d_nd_from_shape(
                        data_index + data_start,
                        self.shape(),
                    )) = my_chunk.data[data_index];
                }
            }

            chunk_index += 1;
        }
    }

    /// Chunked componentwise multiplication into array.
    pub fn cmp_mult_into_chunked<
        ArrayImplOther: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const N: usize,
    >(
        &mut self,
        other: Array<Item, ArrayImplOther, NDIM>,
    ) where
        Self: ChunkedAccess<N, Item = Item>,
    {
        assert_eq!(self.shape(), other.shape());

        let mut chunk_index = 0;

        while let (Some(mut my_chunk), Some(chunk)) =
            (self.get_chunk(chunk_index), other.get_chunk(chunk_index))
        {
            let data_start = chunk.start_index;

            for data_index in 0..chunk.valid_entries {
                my_chunk.data[data_index] *= chunk.data[data_index];
            }

            for data_index in 0..chunk.valid_entries {
                unsafe {
                    *self.get_unchecked_mut(convert_1d_nd_from_shape(
                        data_index + data_start,
                        self.shape(),
                    )) = my_chunk.data[data_index];
                }
            }

            chunk_index += 1;
        }
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Return true of array is empty (that is one dimension is zero), otherwise false.
    pub fn is_empty(&self) -> bool {
        self.shape().iter().copied().min().unwrap() == 0
    }

    /// Return the trace of an array.
    pub fn trace(self) -> Item {
        let k = *self.shape().iter().min().unwrap();

        (0..k).fold(<Item as Zero>::zero(), |acc, index| {
            acc + self.get_value([index; NDIM]).unwrap()
        })
    }

    /// Return the sum of the elements.
    pub fn sum(self) -> Item {
        self.iter().sum()
    }
}

impl<Item: RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>>
    Array<Item, ArrayImpl, 1>
where
    Item::Real: num::Float,
{
    /// Compute the inner product between two vectors.
    ///
    /// The inner product takes the complex conjugate of the `other` argument.
    pub fn inner<ArrayImplOther: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>>(
        &self,
        other: Array<Item, ArrayImplOther, 1>,
    ) -> Item {
        assert_eq!(
            self.number_of_elements(),
            other.number_of_elements(),
            "Arrays must have the same length"
        );

        self.iter()
            .zip(other.iter())
            .fold(<Item as Zero>::zero(), |acc, (elem1, elem_other)| {
                acc + elem1 * elem_other.conj()
            })
    }

    /// Compute the maximum (or inf) norm of a vector.
    pub fn norm_inf(self) -> <Item as RlstScalar>::Real {
        self.iter()
            .map(|elem| <Item as RlstScalar>::abs(elem))
            .reduce(<<Item as RlstScalar>::Real as num::Float>::max)
            .unwrap()
    }

    /// Compute the 1-norm of a vector.
    pub fn norm_1(self) -> <Item as RlstScalar>::Real {
        self.iter()
            .map(|elem| <Item as RlstScalar>::abs(elem))
            .fold(<<Item as RlstScalar>::Real as Zero>::zero(), |acc, elem| {
                acc + elem
            })
    }

    /// Compute the 2-norm of a vector.
    pub fn norm_2(self) -> <Item as RlstScalar>::Real {
        RlstScalar::sqrt(
            self.iter()
                .map(|elem| <Item as RlstScalar>::abs(elem))
                .map(|elem| elem * elem)
                .fold(<<Item as RlstScalar>::Real as Zero>::zero(), |acc, elem| {
                    acc + elem
                }),
        )
    }

    /// Return the length of a 1-dimensional vector.
    pub fn len(&self) -> usize {
        self.shape()[0]
    }

    /// Compute the cross product with vector `other` and store into `res`.
    pub fn cross<
        ArrayImplOther: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImplRes: UnsafeRandomAccessByValue<1, Item = Item>
            + UnsafeRandomAccessMut<1, Item = Item>
            + UnsafeRandomAccessByRef<1, Item = Item>
            + Shape<1>,
    >(
        &self,
        other: Array<Item, ArrayImplOther, 1>,
        mut res: Array<Item, ArrayImplRes, 1>,
    ) {
        assert_eq!(self.len(), 3);
        assert_eq!(other.len(), 3);
        assert_eq!(res.len(), 3);

        let a0 = self.get_value([0]).unwrap();
        let a1 = self.get_value([1]).unwrap();
        let a2 = self.get_value([2]).unwrap();

        let b0 = other.get_value([0]).unwrap();
        let b1 = other.get_value([1]).unwrap();
        let b2 = other.get_value([2]).unwrap();

        res[[0]] = a1 * b2 - a2 * b1;
        res[[1]] = a2 * b0 - a0 * b2;
        res[[2]] = a0 * b1 - a1 * b0;
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
    > Array<Item, ArrayImpl, 2>
where
    Self: MatrixSvd<Item = Item>,
{
    /// Compute the 2-norm of a matrix.
    ///
    /// This method allocates temporary memory during execution.
    pub fn norm_2_alloc(self) -> RlstResult<<Item as RlstScalar>::Real> {
        let k = *self.shape().iter().min().unwrap();

        let mut singular_values = vec![<<Item as RlstScalar>::Real as Zero>::zero(); k];

        self.into_singular_values_alloc(singular_values.as_mut_slice())?;

        Ok(singular_values[0])
    }
}

impl<Item: RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>>
    Array<Item, ArrayImpl, 2>
{
    /// Compute the Frobenius-norm of a matrix.
    pub fn norm_fro(self) -> Item::Real {
        RlstScalar::sqrt(
            self.iter()
                .map(|elem| <Item as RlstScalar>::abs(elem))
                .map(|elem| elem * elem)
                .fold(<<Item as RlstScalar>::Real as Zero>::zero(), |acc, elem| {
                    acc + elem
                }),
        )
    }

    /// Compute the inf-norm of a matrix.
    pub fn norm_inf(self) -> Item::Real {
        self.row_iter()
            .map(|row| row.norm_1())
            .reduce(<<Item as RlstScalar>::Real as num::Float>::max)
            .unwrap()
    }

    /// Compute the 1-norm of a matrix.
    pub fn norm_1(self) -> Item::Real {
        self.col_iter()
            .map(|row| row.norm_1())
            .reduce(<<Item as RlstScalar>::Real as num::Float>::max)
            .unwrap()
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
    > Array<Item, ArrayImpl, 2>
where
    linalg::lu::LuDecomposition<Item, ArrayImpl>:
        linalg::lu::MatrixLuDecomposition<Item = Item, ArrayImpl = ArrayImpl>,
{
    /// Solve a linear system with a single right-hand side.
    ///
    /// The array is overwritten with the LU Decomposition and the right-hand side
    /// is overwritten with the solution. The solution is also returned.
    pub fn into_solve_vec<
        ArrayImplMut: RawAccessMut<Item = Item>
            + UnsafeRandomAccessByValue<1, Item = Item>
            + UnsafeRandomAccessMut<1, Item = Item>
            + Shape<1>
            + Stride<1>,
    >(
        self,
        trans: TransMode,
        mut rhs: Array<Item, ArrayImplMut, 1>,
    ) -> RlstResult<Array<Item, ArrayImplMut, 1>> {
        use linalg::lu::MatrixLuDecomposition;

        let ludecomp = linalg::lu::LuDecomposition::<Item, ArrayImpl>::new(self)?;
        ludecomp.solve_vec(trans, rhs.view_mut())?;
        Ok(rhs)
    }

    /// Compute the determinant of a matrix.
    ///
    /// The array is overwritten by the determinant computation.
    pub fn into_det<
        ArrayImplMut: RawAccessMut<Item = Item>
            + UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>,
    >(
        self,
    ) -> RlstResult<Item> {
        use linalg::lu::MatrixLuDecomposition;

        let ludecomp = linalg::lu::LuDecomposition::<Item, ArrayImpl>::new(self)?;
        Ok(ludecomp.det())
    }

    /// Solve a linear system with multiple right-hand sides.
    ///
    /// The array is overwritten with the LU Decomposition and the right-hand side
    /// is overwritten with the solution. The solution is also returned.
    pub fn into_solve_mat<
        ArrayImplMut: RawAccessMut<Item = Item>
            + UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>,
    >(
        self,
        trans: TransMode,
        mut rhs: Array<Item, ArrayImplMut, 2>,
    ) -> RlstResult<Array<Item, ArrayImplMut, 2>> {
        use linalg::lu::MatrixLuDecomposition;

        let ludecomp = linalg::lu::LuDecomposition::<Item, ArrayImpl>::new(self)?;
        ludecomp.solve_mat(trans, rhs.view_mut())?;
        Ok(rhs)
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>,
    > Array<Item, ArrayImpl, 2>
{
    /// Update self += u * v^T, with u and v vectors.
    pub fn rank1_sum_inplace<
        ArrayImpl1: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImpl2: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    >(
        &mut self,
        u: Array<Item, ArrayImpl1, 1>,
        v: Array<Item, ArrayImpl2, 1>,
    ) {
        assert_eq!(self.shape()[0], u.shape()[0]);
        assert_eq!(self.shape()[1], v.shape()[0]);

        for (mut col, v_elem) in itertools::izip!(self.col_iter_mut(), v.iter()) {
            for (elem, u_elem) in itertools::izip!(col.iter_mut(), u.iter()) {
                *elem += v_elem * u_elem;
            }
        }
    }

    /// Update self *= u * v^T, with u and v vectors.
    pub fn rank1_cmp_product_inplace<
        ArrayImpl1: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
        ArrayImpl2: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    >(
        &mut self,
        u: Array<Item, ArrayImpl1, 1>,
        v: Array<Item, ArrayImpl2, 1>,
    ) {
        assert_eq!(self.shape()[0], u.shape()[0]);
        assert_eq!(self.shape()[1], v.shape()[0]);

        for (mut col, v_elem) in itertools::izip!(self.col_iter_mut(), v.iter()) {
            for (elem, u_elem) in itertools::izip!(col.iter_mut(), u.iter()) {
                *elem *= v_elem * u_elem;
            }
        }
    }
}
