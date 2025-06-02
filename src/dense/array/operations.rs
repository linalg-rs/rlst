//! Operations on arrays.
use std::iter::Sum;
use std::ops::{AddAssign, MulAssign};

use itertools::izip;
use num::traits::{MulAdd, MulAddAssign};
//use crate::{dense::types::RlstResult, TransMode};

use super::iterators::{ArrayDiagIterator, ArrayDiagIteratorMut};
use super::{Array, Shape, UnsafeRandomAccessByValue, UnsafeRandomAccessMut};
use crate::dense::traits::{
    Abs, AbsSquare, ArrayIterator, ArrayIteratorMut, CmpMulAddFrom, CmpMulFrom, Conj, FillFrom,
    FillFromResize, FillWithValue, GetDiag, GetDiagMut, Inner, Len, Max, NormSup, NormTwo,
    ResizeInPlace, Sqrt, SumFrom, Trace, UnsafeRandom1DAccessMut,
};

impl<Item, ArrayImpl, const NDIM: usize> GetDiag for Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
{
    type Item = Item;

    type Iter<'a>
        = ArrayDiagIterator<'a, ArrayImpl, NDIM>
    where
        Self: 'a;

    fn diag_iter(&self) -> Self::Iter<'_> {
        ArrayDiagIterator::new(self)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> GetDiagMut for Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM, Item = Item> + Shape<NDIM>,
{
    type Item = Item;

    type Iter<'a>
        = ArrayDiagIteratorMut<'a, ArrayImpl, NDIM>
    where
        Self: 'a;

    fn diag_iter_mut(&mut self) -> Self::Iter<'_> {
        ArrayDiagIteratorMut::new(self)
    }
}

impl<Item, ArrayImpl, ArrayImplOther, const NDIM: usize> FillFrom<Array<ArrayImplOther, NDIM>>
    for Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item> + Shape<NDIM>,
    Self: ArrayIteratorMut<Item = Item>,
    Array<ArrayImplOther, NDIM>: ArrayIterator<Item = Item> + Shape<NDIM>,
{
    fn fill_from(&mut self, other: &Array<ArrayImplOther, NDIM>) {
        assert_eq!(self.shape(), other.shape());

        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item = other_item;
        }
    }
}

impl<Item, ArrayImpl, ArrayImplOther, const NDIM: usize> SumFrom<Array<ArrayImplOther, NDIM>>
    for Array<ArrayImpl, NDIM>
where
    Item: AddAssign<Item>,
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item> + Shape<NDIM>,
    Self: ArrayIteratorMut<Item = Item>,
    Array<ArrayImplOther, NDIM>: ArrayIterator<Item = Item> + Shape<NDIM>,
{
    fn sum_from(&mut self, other: &Array<ArrayImplOther, NDIM>) {
        assert_eq!(self.shape(), other.shape());

        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item += other_item;
        }
    }
}

impl<Item, ArrayImpl, ArrayImplOther, const NDIM: usize> CmpMulFrom<Array<ArrayImplOther, NDIM>>
    for Array<ArrayImpl, NDIM>
where
    Item: MulAssign<Item>,
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item> + Shape<NDIM>,
    Self: ArrayIteratorMut<Item = Item>,
    Array<ArrayImplOther, NDIM>: ArrayIterator<Item = Item> + Shape<NDIM>,
{
    fn cmp_mult_from(&mut self, other: &Array<ArrayImplOther, NDIM>) {
        assert_eq!(self.shape(), other.shape());

        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item *= other_item;
        }
    }
}

impl<Item, ArrayImpl, ArrayImplOther1, ArrayImplOther2, const NDIM: usize>
    CmpMulAddFrom<Array<ArrayImplOther1, NDIM>, Array<ArrayImplOther2, NDIM>>
    for Array<ArrayImpl, NDIM>
where
    Item: MulAddAssign<Item>,
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item> + Shape<NDIM>,
    Self: ArrayIteratorMut<Item = Item>,
    Array<ArrayImplOther1, NDIM>: ArrayIterator<Item = Item> + Shape<NDIM>,
    Array<ArrayImplOther2, NDIM>: ArrayIterator<Item = Item> + Shape<NDIM>,
{
    fn cmp_mul_add_from(
        &mut self,
        other1: &Array<ArrayImplOther1, NDIM>,
        other2: &Array<ArrayImplOther2, NDIM>,
    ) {
        assert_eq!(self.shape(), other1.shape());
        assert_eq!(self.shape(), other2.shape());

        for (item, other_item1, other_item2) in izip!(self.iter_mut(), other1.iter(), other2.iter())
        {
            MulAddAssign::mul_add_assign(item, other_item1, other_item2);
        }
    }
}

impl<ArrayImpl, ArrayImplOther, const NDIM: usize> FillFromResize<Array<ArrayImplOther, NDIM>>
    for Array<ArrayImpl, NDIM>
where
    ArrayImpl: ResizeInPlace<NDIM> + Shape<NDIM>,
    Self: FillFrom<Array<ArrayImplOther, NDIM>>,
    ArrayImplOther: Shape<NDIM>,
{
    fn fill_from_resize(&mut self, other: &Array<ArrayImplOther, NDIM>) {
        self.resize_in_place(other.shape());
        self.fill_from(other);
    }
}

impl<Item, ArrayImpl, const NDIM: usize> FillWithValue for Array<ArrayImpl, NDIM>
where
    Array<ArrayImpl, NDIM>: ArrayIteratorMut<Item = Item>,
    Item: Copy,
{
    type Item = Item;

    fn fill_with_value(&mut self, value: Item) {
        for item in self.iter_mut() {
            *item = value;
        }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Trace for Array<ArrayImpl, NDIM>
where
    Array<ArrayImpl, NDIM>: GetDiag<Item = Item>,
    Item: std::iter::Sum,
{
    type Item = Item;

    fn trace(&self) -> Self::Item {
        self.diag_iter().sum::<Self::Item>()
    }
}

impl<Item, ArrayImpl, ArrayImplOther> Inner<Array<ArrayImplOther, 1>> for Array<ArrayImpl, 1>
where
    Item: Default + MulAdd<Output = Item> + Conj<Output = Item>,
    Self: ArrayIterator<Item = Item> + Len,
    Array<ArrayImplOther, 1>: ArrayIterator<Item = Item> + Len,
{
    type Item = Item;

    fn inner(&self, other: &Array<ArrayImplOther, 1>) -> Self::Item {
        assert_eq!(self.len(), other.len());
        izip!(self.iter(), other.iter()).fold(Default::default(), |acc, (elem1, elem2)| {
            elem1.mul_add(elem2.conj(), acc)
        })
    }
}

impl<ArrayImpl> NormSup for Array<ArrayImpl, 1>
where
    Self: ArrayIterator,
    <Self as ArrayIterator>::Item: Abs,
    <<Self as ArrayIterator>::Item as Abs>::Output:
        Max<Output = <<Self as ArrayIterator>::Item as Abs>::Output> + Default,
{
    type Item = <<Self as ArrayIterator>::Item as Abs>::Output;

    fn norm_sup(&self) -> Self::Item {
        self.iter().fold(
            <<<Self as ArrayIterator>::Item as Abs>::Output as Default>::default(),
            |acc, elem| Max::max(&acc, &elem.abs()),
        )
    }
}

impl<ArrayImpl> AbsSquare for Array<ArrayImpl, 1>
where
    Self: ArrayIterator,
    <Self as ArrayIterator>::Item: AbsSquare,
    <<Self as ArrayIterator>::Item as AbsSquare>::Output: Sum,
{
    type Output = <<Self as ArrayIterator>::Item as AbsSquare>::Output;

    fn abs_square(&self) -> Self::Output {
        self.iter().map(|elem| elem.abs_square()).sum()
    }
}

impl<ArrayImpl> NormTwo for Array<ArrayImpl, 1>
where
    Array<ArrayImpl, 1>: AbsSquare,
    <Array<ArrayImpl, 1> as AbsSquare>::Output: Sqrt,
{
    type Item = <<Self as AbsSquare>::Output as Sqrt>::Output;

    fn norm_2(&self) -> Self::Item {
        self.abs_square().sqrt()
    }
}

// /// Compute the maximum (or inf) norm of a vector.
// pub fn norm_inf(self) -> <Item as RlstScalar>::Real {
//     self.iter()
//         .map(|elem| <Item as RlstScalar>::abs(elem))
//         .reduce(<<Item as RlstScalar>::Real as num::Float>::max)
//         .unwrap()
// }
//
//     /// Compute the 1-norm of a vector.
//     pub fn norm_1(self) -> <Item as RlstScalar>::Real {
//         self.iter()
//             .map(|elem| <Item as RlstScalar>::abs(elem))
//             .fold(<<Item as RlstScalar>::Real as Zero>::zero(), |acc, elem| {
//                 acc + elem
//             })
//     }

//     /// Compute the 2-norm of a vector.
//     pub fn norm_2(self) -> <Item as RlstScalar>::Real {
//         RlstScalar::sqrt(
//             self.iter()
//                 .map(|elem| <Item as RlstScalar>::abs(elem))
//                 .map(|elem| elem * elem)
//                 .fold(<<Item as RlstScalar>::Real as Zero>::zero(), |acc, elem| {
//                     acc + elem
//                 }),
//         )
//     }

//     /// Compute the cross product with vector `other` and store into `res`.
//     pub fn cross<
//         ArrayImplOther: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
//         ArrayImplRes: UnsafeRandomAccessByValue<1, Item = Item>
//             + UnsafeRandomAccessMut<1, Item = Item>
//             + UnsafeRandomAccessByRef<1, Item = Item>
//             + Shape<1>,
//     >(
//         &self,
//         other: Array<Item, ArrayImplOther, 1>,
//         mut res: Array<Item, ArrayImplRes, 1>,
//     ) {
//         assert_eq!(self.len(), 3);
//         assert_eq!(other.len(), 3);
//         assert_eq!(res.len(), 3);

//         let a0 = self.get_value([0]).unwrap();
//         let a1 = self.get_value([1]).unwrap();
//         let a2 = self.get_value([2]).unwrap();

//         let b0 = other.get_value([0]).unwrap();
//         let b1 = other.get_value([1]).unwrap();
//         let b2 = other.get_value([2]).unwrap();

//         res[[0]] = a1 * b2 - a2 * b1;
//         res[[1]] = a2 * b0 - a0 * b2;
//         res[[2]] = a0 * b1 - a1 * b0;
//     }
// }

// impl<
//         Item: RlstScalar,
//         ArrayImpl: UnsafeRandomAccessByValue<1, Item = Item>
//             + Shape<1>
//             + UnsafeRandom1DAccessByValue<Item = Item>,
//     > Array<Item, ArrayImpl, 1>
// where
//     Item::Real: num::Float,
// {
// }

// // impl<
// //         Item: RlstScalar,
// //         ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
// //             + Shape<2>
// //             + Stride<2>
// //             + RawAccessMut<Item = Item>,
// //     > Array<Item, ArrayImpl, 2>
// // where
// //     Item: MatrixSvd,
// // {
// //     /// Compute the 2-norm of a matrix.
// //     ///
// //     /// This method allocates temporary memory during execution.
// //     pub fn norm_2_alloc(self) -> RlstResult<<Item as RlstScalar>::Real> {
// //         let k = *self.shape().iter().min().unwrap();

// //         let mut singular_values = vec![<<Item as RlstScalar>::Real as Zero>::zero(); k];

// //         self.into_singular_values_alloc(singular_values.as_mut_slice())?;

// //         Ok(singular_values[0])
// //     }
// // }

// // impl<
// //         Item: RlstScalar,
// //         ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
// //             + Shape<2>
// //             + UnsafeRandom1DAccessByValue<Item = Item>,
// //     > Array<Item, ArrayImpl, 2>
// // {
// //     /// Compute the Frobenius-norm of a matrix.
// //     pub fn norm_fro(self) -> Item::Real {
// //         RlstScalar::sqrt(
// //             self.iter()
// //                 .map(|elem| <Item as RlstScalar>::abs(elem))
// //                 .map(|elem| elem * elem)
// //                 .fold(<<Item as RlstScalar>::Real as Zero>::zero(), |acc, elem| {
// //                     acc + elem
// //                 }),
// //         )
// //     }

// //     /// Compute the inf-norm of a matrix.
// //     pub fn norm_inf(self) -> Item::Real {
// //         self.row_iter()
// //             .map(|row| row.norm_1())
// //             .reduce(<<Item as RlstScalar>::Real as num::Float>::max)
// //             .unwrap()
// //     }

// //     /// Compute the 1-norm of a matrix.
// //     pub fn norm_1(self) -> Item::Real {
// //         self.col_iter()
// //             .map(|row| row.norm_1())
// //             .reduce(<<Item as RlstScalar>::Real as num::Float>::max)
// //             .unwrap()
// //     }
// // }

// // impl<
// //         Item: RlstScalar,
// //         ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
// //             + Shape<2>
// //             + Stride<2>
// //             + RawAccessMut<Item = Item>,
// //     > Array<Item, ArrayImpl, 2>
// // where
// //     linalg::lu::LuDecomposition<Item, ArrayImpl>:
// //         linalg::lu::MatrixLuDecomposition<Item = Item, ArrayImpl = ArrayImpl>,
// // {
// //     /// Solve a linear system with a single right-hand side.
// //     ///
// //     /// The array is overwritten with the LU Decomposition and the right-hand side
// //     /// is overwritten with the solution. The solution is also returned.
// //     pub fn into_solve_vec<
// //         ArrayImplMut: RawAccessMut<Item = Item>
// //             + UnsafeRandomAccessByValue<1, Item = Item>
// //             + UnsafeRandomAccessMut<1, Item = Item>
// //             + Shape<1>
// //             + Stride<1>,
// //     >(
// //         self,
// //         trans: TransMode,
// //         mut rhs: Array<Item, ArrayImplMut, 1>,
// //     ) -> RlstResult<Array<Item, ArrayImplMut, 1>> {
// //         use linalg::lu::MatrixLuDecomposition;

// //         let ludecomp = linalg::lu::LuDecomposition::<Item, ArrayImpl>::new(self)?;
// //         ludecomp.solve_vec(trans, rhs.r_mut())?;
// //         Ok(rhs)
// //     }

// //     /// Compute the determinant of a matrix.
// //     ///
// //     /// The array is overwritten by the determinant computation.
// //     pub fn into_det<
// //         ArrayImplMut: RawAccessMut<Item = Item>
// //             + UnsafeRandomAccessByValue<2, Item = Item>
// //             + UnsafeRandomAccessMut<2, Item = Item>
// //             + Shape<2>
// //             + Stride<2>,
// //     >(
// //         self,
// //     ) -> RlstResult<Item> {
// //         use linalg::lu::MatrixLuDecomposition;

// //         let ludecomp = linalg::lu::LuDecomposition::<Item, ArrayImpl>::new(self)?;
// //         Ok(ludecomp.det())
// //     }

// //     /// Solve a linear system with multiple right-hand sides.
// //     ///
// //     /// The array is overwritten with the LU Decomposition and the right-hand side
// //     /// is overwritten with the solution. The solution is also returned.
// //     pub fn into_solve_mat<
// //         ArrayImplMut: RawAccessMut<Item = Item>
// //             + UnsafeRandomAccessByValue<2, Item = Item>
// //             + UnsafeRandomAccessMut<2, Item = Item>
// //             + Shape<2>
// //             + Stride<2>,
// //     >(
// //         self,
// //         trans: TransMode,
// //         mut rhs: Array<Item, ArrayImplMut, 2>,
// //     ) -> RlstResult<Array<Item, ArrayImplMut, 2>> {
// //         use linalg::lu::MatrixLuDecomposition;

// //         let ludecomp = linalg::lu::LuDecomposition::<Item, ArrayImpl>::new(self)?;
// //         ludecomp.solve_mat(trans, rhs.r_mut())?;
// //         Ok(rhs)
// //     }
// // }

// #[cfg(test)]
// mod test {
//     use crate::dense::{array::empty_array, traits::Conj};

// fn op_test() {
//     let mut a = empty_array::<f64, _>();
//
//     let fun = |a: f64| -> f64 { a };
//
//     a.apply_unary_op(Conj::conj);
//     // }
// }
