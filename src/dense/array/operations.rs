//! Operations on arrays.
use std::iter::Sum;
use std::ops::{AddAssign, MulAssign};

use itertools::izip;
use num::traits::{MulAdd, MulAddAssign};
//use crate::{dense::types::RlstResult, TransMode};

use crate::traits::accessors::UnsafeRandom1DAccessByValue;
use crate::traits::iterators::GetDiagByRef;
use crate::traits::iterators::{ColumnIterator, ColumnIteratorMut};
use crate::traits::{
    accessors::{
        UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef,
        UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
    },
    array::{
        BaseItem, CmpMulAddFrom, CmpMulFrom, ConjArray, EvaluateArray, FillFrom, FillFromResize,
        FillWithValue, Len, ResizeInPlace, Shape, SumFrom, ToType, Trace,
    },
    iterators::{
        AijIteratorByRef, AijIteratorByValue, AijIteratorMut, ArrayIteratorByRef,
        ArrayIteratorByValue, ArrayIteratorMut, GetDiagByValue, GetDiagMut,
    },
    linalg::base::{Inner, NormSup, NormTwo},
    number_traits::{Abs, AbsSquare, Conj, Max, Sqrt},
};
use crate::{
    AsMultiIndex, ContainerTypeHint, DispatchEval, DispatchEvalRowMajor, EvaluateRowMajorArray,
    FillFromIter, NormOne,
};

use super::iterators::{
    ArrayDefaultIteratorByRef, ArrayDefaultIteratorByValue, ArrayDefaultIteratorMut,
    ArrayDiagIteratorByRef, ArrayDiagIteratorByValue, ArrayDiagIteratorMut, ColIterator,
    ColIteratorMut, MultiIndexIterator,
};
use super::operators::unary_op::ArrayUnaryOperator;
use super::reference::{ArrayRef, ArrayRefMut};
use super::slice::ArraySlice;
use super::{Array, EvalDispatcher, EvalRowMajorDispatcher};

impl<Item, ArrayImpl, const NDIM: usize> GetDiagByValue for Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
{
    type Iter<'a>
        = ArrayDiagIteratorByValue<'a, ArrayImpl, NDIM>
    where
        Self: 'a;

    fn diag_iter_value(&self) -> Self::Iter<'_> {
        ArrayDiagIteratorByValue::new(self)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> GetDiagByRef for Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM, Item = Item> + Shape<NDIM>,
{
    type Iter<'a>
        = ArrayDiagIteratorByRef<'a, ArrayImpl, NDIM>
    where
        Self: 'a;

    fn diag_iter_ref(&self) -> Self::Iter<'_> {
        ArrayDiagIteratorByRef::new(self)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> GetDiagMut for Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM, Item = Item> + Shape<NDIM>,
{
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
    Array<ArrayImplOther, NDIM>: ArrayIteratorByValue<Item = Item> + Shape<NDIM>,
{
    fn fill_from(&mut self, other: &Array<ArrayImplOther, NDIM>) {
        assert_eq!(self.shape(), other.shape());

        for (item, other_item) in self.iter_mut().zip(other.iter_value()) {
            *item = other_item;
        }
    }
}

impl<ArrayImpl, Iter: Iterator<Item = ArrayImpl::Item>, const NDIM: usize> FillFromIter<Iter>
    for Array<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem,
    Array<ArrayImpl, NDIM>: ArrayIteratorMut<Item = ArrayImpl::Item>,
{
    fn fill_from_iter(&mut self, iter: Iter) {
        for (item, other_item) in izip!(self.iter_mut(), iter) {
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
    Array<ArrayImplOther, NDIM>: ArrayIteratorByValue<Item = Item> + Shape<NDIM>,
{
    fn sum_from(&mut self, other: &Array<ArrayImplOther, NDIM>) {
        assert_eq!(self.shape(), other.shape());

        for (item, other_item) in self.iter_mut().zip(other.iter_value()) {
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
    Array<ArrayImplOther, NDIM>: ArrayIteratorByValue<Item = Item> + Shape<NDIM>,
{
    fn cmp_mul_from(&mut self, other: &Array<ArrayImplOther, NDIM>) {
        assert_eq!(self.shape(), other.shape());

        for (item, other_item) in self.iter_mut().zip(other.iter_value()) {
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
    Array<ArrayImplOther1, NDIM>: ArrayIteratorByValue<Item = Item> + Shape<NDIM>,
    Array<ArrayImplOther2, NDIM>: ArrayIteratorByValue<Item = Item> + Shape<NDIM>,
{
    fn cmp_mul_add_from(
        &mut self,
        other1: &Array<ArrayImplOther1, NDIM>,
        other2: &Array<ArrayImplOther2, NDIM>,
    ) {
        assert_eq!(self.shape(), other1.shape());
        assert_eq!(self.shape(), other2.shape());

        for (item, other_item1, other_item2) in
            izip!(self.iter_mut(), other1.iter_value(), other2.iter_value())
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
    fn fill_with_value(&mut self, value: Item) {
        for item in self.iter_mut() {
            *item = value;
        }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Trace for Array<ArrayImpl, NDIM>
where
    Array<ArrayImpl, NDIM>: GetDiagByValue<Item = Item>,
    Item: std::iter::Sum,
{
    fn trace(&self) -> Self::Item {
        self.diag_iter_value().sum::<Self::Item>()
    }
}

impl<Item, ArrayImpl, ArrayImplOther> Inner<Array<ArrayImplOther, 1>> for Array<ArrayImpl, 1>
where
    Item: Default + MulAdd<Output = Item> + Conj<Output = Item>,
    ArrayImpl: BaseItem<Item = Item>,
    ArrayImplOther: BaseItem<Item = Item>,
    Self: ArrayIteratorByValue<Item = Item> + Len,
    Array<ArrayImplOther, 1>: ArrayIteratorByValue<Item = Item> + Len,
{
    type Output = Item;

    fn inner(&self, other: &Array<ArrayImplOther, 1>) -> Self::Output {
        assert_eq!(self.len(), other.len());
        izip!(self.iter_value(), other.iter_value())
            .fold(Default::default(), |acc, (elem, other_elem)| {
                MulAdd::mul_add(elem, Conj::conj(&other_elem), acc)
            })
    }
}

impl<ArrayImpl> NormSup for Array<ArrayImpl, 1>
where
    Self: ArrayIteratorByValue,
    <Self as BaseItem>::Item: Abs,
    <<Self as BaseItem>::Item as Abs>::Output:
        Max<Output = <<Self as BaseItem>::Item as Abs>::Output> + Default,
{
    type Output = <<Self as BaseItem>::Item as Abs>::Output;

    fn norm_sup(&self) -> Self::Output {
        self.iter_value()
            .fold(<Self::Output as Default>::default(), |acc, elem| {
                Max::max(&acc, &elem.abs())
            })
    }
}

impl<ArrayImpl> AbsSquare for Array<ArrayImpl, 1>
where
    Self: ArrayIteratorByValue,
    <Self as BaseItem>::Item: AbsSquare,
    <<Self as BaseItem>::Item as AbsSquare>::Output: Sum,
{
    type Output = <<Self as BaseItem>::Item as AbsSquare>::Output;

    fn abs_square(&self) -> Self::Output {
        self.iter_value().map(|elem| elem.abs_square()).sum()
    }
}

impl<ArrayImpl> NormOne for Array<ArrayImpl, 1>
where
    ArrayImpl: BaseItem,
    ArrayImpl::Item: Abs,
    <ArrayImpl::Item as Abs>::Output:
        std::ops::Add<Output = <ArrayImpl::Item as Abs>::Output> + Default,
    Self: ArrayIteratorByValue<Item = ArrayImpl::Item>,
{
    type Output = <<Self as BaseItem>::Item as Abs>::Output;

    fn norm_1(&self) -> Self::Output {
        self.iter_value()
            .fold(<Self::Output as Default>::default(), |acc, elem| {
                acc + elem.abs()
            })
    }
}

impl<ArrayImpl> NormTwo for Array<ArrayImpl, 1>
where
    Array<ArrayImpl, 1>: AbsSquare,
    <Array<ArrayImpl, 1> as AbsSquare>::Output: Sqrt,
{
    type Output = <<Self as AbsSquare>::Output as Sqrt>::Output;

    fn norm_2(&self) -> Self::Output {
        self.abs_square().sqrt()
    }
}

impl<Item, ArrayImpl, const NDIM: usize> ConjArray for Array<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem<Item = Item>,
    Item: Conj,
{
    type Output = Array<
        ArrayUnaryOperator<
            Item,
            <Item as Conj>::Output,
            ArrayImpl,
            fn(Item) -> <Item as Conj>::Output,
            NDIM,
        >,
        NDIM,
    >;

    fn conj(self) -> Self::Output {
        fn conj<Item: Conj>(item: Item) -> <Item as Conj>::Output {
            item.conj()
        }
        Array::new(ArrayUnaryOperator::new(self, conj))
    }
}

// impl<Item, ArrayImpl, const NDIM: usize> EvaluateArray for Array<ArrayImpl, NDIM>
// where
//     DynArray<Item, NDIM>: FillFromResize<Array<ArrayImpl, NDIM>>,
//     Item: Clone + Default,
//     ArrayImpl: BaseItem<Item = Item>,
// {
//     type Output = DynArray<Item, NDIM>;
//
//     fn eval(&self) -> Self::Output {
//         DynArray::new_from(self)
//     }
// }

impl<Item, ArrayImpl, const NDIM: usize> EvaluateArray for Array<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerTypeHint + UnsafeRandom1DAccessByValue<Item = Item>,
    EvalDispatcher<ArrayImpl::TypeHint, ArrayImpl>: DispatchEval<NDIM, ArrayImpl = ArrayImpl>,
{
    type Output = <EvalDispatcher<ArrayImpl::TypeHint, ArrayImpl> as DispatchEval<NDIM>>::Output;

    fn eval(&self) -> Self::Output {
        let dispatcher = EvalDispatcher::<ArrayImpl::TypeHint, ArrayImpl>::default();
        dispatcher.dispatch(self)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> EvaluateRowMajorArray for Array<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerTypeHint + UnsafeRandom1DAccessByValue<Item = Item>,
    EvalRowMajorDispatcher<ArrayImpl::TypeHint, ArrayImpl>:
        DispatchEvalRowMajor<NDIM, ArrayImpl = ArrayImpl>,
{
    type Output = <EvalRowMajorDispatcher<ArrayImpl::TypeHint, ArrayImpl> as DispatchEvalRowMajor<NDIM>>::Output;

    fn eval_row_major(&self) -> Self::Output {
        let dispatcher = EvalRowMajorDispatcher::<ArrayImpl::TypeHint, ArrayImpl>::default();
        dispatcher.dispatch(self)
    }
}

impl<Item, T, ArrayImpl, const NDIM: usize> ToType<T> for Array<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem<Item = Item>,
{
    type Item = Item;

    type Output = Array<ArrayUnaryOperator<Item, T, ArrayImpl, fn(Item) -> T, NDIM>, NDIM>;

    fn into_type(self) -> Array<ArrayUnaryOperator<Item, T, ArrayImpl, fn(Item) -> T, NDIM>, NDIM>
    where
        Item: Into<T>,
    {
        Array::new(ArrayUnaryOperator::new(self, |item| item.into()))
    }
}

impl<ArrayImpl, const NDIM: usize> ArrayIteratorByValue for Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<NDIM>,
{
    type Iter<'a>
        = ArrayDefaultIteratorByValue<'a, ArrayImpl, NDIM>
    where
        Self: 'a;

    fn iter_value(&self) -> Self::Iter<'_> {
        ArrayDefaultIteratorByValue::new(self)
    }
}

impl<ArrayImpl, const NDIM: usize> ArrayIteratorByRef for Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByRef + Shape<NDIM>,
{
    type Iter<'a>
        = ArrayDefaultIteratorByRef<'a, ArrayImpl, NDIM>
    where
        Self: 'a;

    fn iter_ref(&self) -> Self::Iter<'_> {
        ArrayDefaultIteratorByRef::new(self)
    }
}

impl<ArrayImpl, const NDIM: usize> ArrayIteratorMut for Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut + Shape<NDIM>,
{
    type IterMut<'a>
        = ArrayDefaultIteratorMut<'a, ArrayImpl, NDIM>
    where
        Self: 'a;

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        ArrayDefaultIteratorMut::new(self)
    }
}

impl<ArrayImpl> ColumnIterator for Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessByValue<2>,
{
    type Col<'a>
        = Array<ArraySlice<ArrayRef<'a, ArrayImpl, 2>, 2, 1>, 1>
    where
        Self: 'a;

    type Iter<'a>
        = ColIterator<'a, ArrayImpl, 2>
    where
        Self: 'a;

    fn col_iter(&self) -> Self::Iter<'_> {
        ColIterator::new(self)
    }
}

impl<ArrayImpl> ColumnIteratorMut for Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessMut<2>,
{
    type Col<'a>
        = Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>
    where
        Self: 'a;

    type Iter<'a>
        = ColIteratorMut<'a, ArrayImpl, 2>
    where
        Self: 'a;

    fn col_iter_mut(&mut self) -> Self::Iter<'_> {
        ColIteratorMut::new(self)
    }
}

impl<ArrayImpl> crate::traits::iterators::RowIterator for Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessByValue<2>,
{
    type Row<'a>
        = Array<ArraySlice<ArrayRef<'a, ArrayImpl, 2>, 2, 1>, 1>
    where
        Self: 'a;

    type Iter<'a>
        = crate::dense::array::iterators::RowIterator<'a, ArrayImpl, 2>
    where
        Self: 'a;

    fn row_iter(&self) -> Self::Iter<'_> {
        crate::dense::array::iterators::RowIterator::new(self)
    }
}

impl<ArrayImpl> crate::traits::iterators::RowIteratorMut for Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessMut<2>,
{
    type Row<'a>
        = Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>
    where
        Self: 'a;

    type Iter<'a>
        = crate::dense::array::iterators::RowIteratorMut<'a, ArrayImpl, 2>
    where
        Self: 'a;

    fn row_iter_mut(&mut self) -> Self::Iter<'_> {
        crate::dense::array::iterators::RowIteratorMut::new(self)
    }
}

impl<ArrayImpl> AijIteratorByValue for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<2>,
{
    type Iter<'a>
        = MultiIndexIterator<std::iter::Enumerate<ArrayDefaultIteratorByValue<'a, ArrayImpl, 2>>, 2>
    where
        Self: 'a;

    fn iter_aij_value(&self) -> Self::Iter<'_> {
        let iter = ArrayDefaultIteratorByValue::new(self);
        AsMultiIndex::multi_index(std::iter::Iterator::enumerate(iter), self.shape())
    }
}

impl<ArrayImpl> AijIteratorByRef for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByRef + Shape<2>,
{
    type Iter<'a>
        = MultiIndexIterator<std::iter::Enumerate<ArrayDefaultIteratorByRef<'a, ArrayImpl, 2>>, 2>
    where
        Self: 'a;

    fn iter_aij_ref(&self) -> Self::Iter<'_> {
        let iter = ArrayDefaultIteratorByRef::new(self);
        AsMultiIndex::multi_index(std::iter::Iterator::enumerate(iter), self.shape())
    }
}

impl<ArrayImpl> AijIteratorMut for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessMut + Shape<2>,
{
    type Iter<'a>
        = MultiIndexIterator<std::iter::Enumerate<ArrayDefaultIteratorMut<'a, ArrayImpl, 2>>, 2>
    where
        Self: 'a;

    fn iter_aij_mut(&mut self) -> Self::Iter<'_> {
        let shape = self.shape();
        let iter = ArrayDefaultIteratorMut::new(self);
        AsMultiIndex::multi_index(std::iter::Iterator::enumerate(iter), shape)
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
