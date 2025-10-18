//! MPI Distributed arrays.
//!
//! A distributed array is distributed across the ranks along the first axis.
//! The data distribution is defined through an [IndexLayout].

use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use std::rc::Rc;

use crate::base_types::{c32, c64};
use crate::dense::array::operators::addition::ArrayAddition;
use crate::dense::array::operators::cast::ArrayCast;
use crate::dense::array::operators::cmp_wise_division::CmpWiseDivision;
use crate::dense::array::operators::cmp_wise_product::CmpWiseProduct;
use crate::dense::array::operators::coerce::CoerceArray;
use crate::dense::array::operators::mul_add::MulAddImpl;
use crate::dense::array::operators::scalar_mult::ArrayScalarMult;
use crate::dense::array::operators::subtraction::ArraySubtraction;
use crate::dense::array::operators::unary_op::ArrayUnaryOperator;
use crate::dense::array::reference::{self, ArrayRef, ArrayRefMut};
use crate::dense::array::slice::ArraySlice;
use crate::dense::base_array::BaseArray;
use crate::dense::data_container::VectorContainer;
use crate::dense::layout::row_major_stride_from_shape;
use crate::distributed_tools::{IndexLayout, scatterv, scatterv_root};
use num::traits::MulAdd;

use crate::dense::array::{DynArray, StridedDynArray, StridedSliceArray};
use crate::{
    Array, BaseItem, Conj, EvaluateObject, Max, RlstResult, Shape, UnsafeRandom1DAccessByValue,
    UnsafeRandom1DAccessMut, UnsafeRandomAccessMut,
};
use crate::{EvaluateRowMajorArray, UnsafeRandomAccessByValue};

use mpi::Rank;
use mpi::datatype::PartitionMut;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence, Root};

/// Distributed Array.
///
/// A distributed array is a an array that is distributed along the first dimension with
/// respect to a given [IndexLayout]. On each process the `local` data is just a standard [Array]
/// and supports all operations on standard arrays.
pub struct DistributedArray<'a, C: Communicator, ArrayImpl, const NDIM: usize> {
    /// The index layout of the vector.
    ///
    /// The index layout is stored via an [Rc] since we often need to a reference to the MPI Communicator
    /// in the [IndexLayout] at the same time as a mutable reference to the `local` data. But this would
    /// be disallowed by Rust's static borrow checker. Hence, runtime borrow checking through an [Rc].
    pub index_layout: Rc<IndexLayout<'a, C>>,
    /// The local data of the vector
    pub local: Array<ArrayImpl, NDIM>,
}

impl<'a, C, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM>,
{
    /// Create a new distributed array.
    ///
    /// It is expected that `arr` has first dimension compatibute with `index_layout` and
    /// that the other dimensions are identical across processes. This is not checked by
    /// `new`. Generally, it is advisable not to use this method to instantiate a new
    /// distributed array but either the [dist_vec] or the [dist_mat] macro that create
    /// distributed vectors or matrices.
    pub fn new(index_layout: Rc<IndexLayout<'a, C>>, arr: Array<ArrayImpl, NDIM>) -> Self {
        let number_of_local_indices = index_layout.number_of_local_indices();
        assert_eq!(
            number_of_local_indices,
            arr.shape()[0],
            "The number of local indices in the index layout ({}) does not match the first dimension of the array ({})",
            number_of_local_indices,
            arr.shape()[0]
        );
        DistributedArray {
            index_layout,
            local: arr,
        }
    }

    /// Get a reference struct to an existing array.
    pub fn r<'b>(&'b self) -> DistributedArray<'a, C, ArrayRef<'b, ArrayImpl, NDIM>, NDIM> {
        DistributedArray::new(self.index_layout.clone(), self.local.r())
    }

    /// Get a mutable reference struct to an existing array.
    pub fn r_mut<'b>(
        &'b mut self,
    ) -> DistributedArray<'a, C, ArrayRefMut<'b, ArrayImpl, NDIM>, NDIM> {
        DistributedArray::new(self.index_layout.clone(), self.local.r_mut())
    }

    /// Check that index layout and shape is the same as the other array.
    pub fn is_compatible_with<ArrayImplOther>(
        &self,
        other: &DistributedArray<'a, C, ArrayImplOther, NDIM>,
    ) -> bool
    where
        C: Communicator,
        ArrayImplOther: Shape<NDIM>,
    {
        self.index_layout.is_same(&other.index_layout) && self.shape() == other.shape()
    }
}

impl<'a, C, Item, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessByValue<Item = Item>,
    Item: Equivalence + Copy + Default,
{
    /// Gather `Self` to all processes and store in `arr`.
    ///
    /// This method collects the distributed array into local arrays on all processes.
    /// The output is a local array on each process that has all the data of the original distributed array.
    pub fn gather_to_all(&self) -> DynArray<Item, NDIM> {
        let comm = self.index_layout.comm();
        let this_process = comm.this_process();

        // We evaluate into a new array to ensure that we are row-major contiguous.
        let send_arr = StridedDynArray::row_major_from(&self.local);

        // The local shape of the array.
        let local_shape = self.local.shape();
        // This is the number of elements from the second dimension onwards.
        // This is the same on every process as only the first dmension is distributed.
        // Since an empty iterator returns a one value this line works also if we have a 1D array.
        let other_dims_count = local_shape.iter().skip(1).product::<usize>();
        // This is going to be the new global shape. All dimensions except the first one are the
        // same as the local shape. The first dimension is the number of global indices.
        let global_shape = {
            let mut tmp = local_shape;
            tmp[0] = self.index_layout.number_of_global_indices();
            tmp
        };
        // We create a new row-major array to send data into.
        // This is necessary as we are sending row slices
        let mut recv_arr = StridedDynArray::<Item, NDIM>::row_major(global_shape);

        // Take the first comm.size elements of the counts. (counts is 1 + comm.size long)
        // We also have to multiply the counts by the number of elements in the second dimension
        // onwards. This is the same on every process as only the first dimension is distributed.
        let counts = self
            .index_layout
            .local_counts()
            .iter()
            .map(|&x| (other_dims_count * x) as i32)
            .collect::<Vec<i32>>();
        let displacements: Vec<i32> = crate::distributed_tools::displacements(&counts);
        let mut partition = PartitionMut::new(recv_arr.data_mut(), counts, displacements);
        this_process.all_gather_varcount_into(send_arr.data(), &mut partition);
        // Finally, we convert the data back to a standard column-major array
        recv_arr.eval()
    }
}

impl<'a, C, ArrayImpl, Item, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessByValue<Item = Item>,
    Item: Equivalence + Copy + Default,
{
    /// Gather the array to rank `root`.
    ///
    /// This method needs to be called on all ranks that are not root. On root call [DistributedArray::gather_to_one_root].
    pub fn gather_to_one(&self, root: usize) {
        let comm = self.index_layout.comm();
        let send_arr = StridedDynArray::row_major_from(&self.local);
        let target_process = comm.process_at_rank(root as Rank);

        target_process.gather_varcount_into(send_arr.data());
    }

    /// Gather the array to a single rank.
    ///
    /// Call this on the `root` to which the array is sent. On other ranks call [DistributedArray::gather_to_one].
    pub fn gather_to_one_root(&self) -> DynArray<Item, NDIM> {
        let comm = self.index_layout.comm();
        let this_process = comm.this_process();

        // We evaluate into a new array to ensure that we are row-major contiguous.
        let send_arr = StridedDynArray::row_major_from(&self.local);

        // The local shape of the array.
        let local_shape = self.local.shape();
        // This is the number of elements from the second dimension onwards.
        // This is the same on every process as only the first dmension is distributed.
        // Since an empty iterator returns a one value this line works also if we have a 1D array.
        let other_dims_count = local_shape.iter().skip(1).product::<usize>();
        // This is going to be the new global shape. All dimensions except the first one are the
        // same as the local shape. The first dimension is the number of global indices.
        let global_shape = {
            let mut tmp = local_shape;
            tmp[0] = self.index_layout.number_of_global_indices();
            tmp
        };
        // We create a new row-major array to send data into.
        // This is necessary as we are sending row slices
        let mut recv_arr = StridedDynArray::<Item, NDIM>::row_major(global_shape);

        // Take the first comm.size elements of the counts. (counts is 1 + comm.size long)
        // We also have to multiply the counts by the number of elements in the second dimension
        // onwards. This is the same on every process as only the first dimension is distributed.
        let counts = self
            .index_layout
            .local_counts()
            .iter()
            .map(|&x| (other_dims_count * x) as i32)
            .collect::<Vec<i32>>();

        let displacements: Vec<i32> = crate::distributed_tools::displacements(&counts);
        let mut partition = PartitionMut::new(recv_arr.data_mut(), counts, displacements);
        this_process.gather_varcount_into_root(send_arr.data(), &mut partition);
        // Finally, we convert the data back to a standard column-major array
        recv_arr.eval()
    }
}

impl<ArrayImpl, Item, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessByValue<Item = Item>,
    Item: Equivalence + Copy + Default,
{
    /// Scatter the array out to all nodes using the given `index_layout`.
    ///
    /// The data is always scattered out along the first axis of the array.
    /// Call this method on root. On other ranks call [Array::scatter_from_one].
    pub fn scatter_from_one_root<'a, C: Communicator>(
        &self,
        index_layout: Rc<IndexLayout<'a, C>>,
    ) -> DistributedArray<'a, C, BaseArray<VectorContainer<Item>, NDIM>, NDIM> {
        let comm = index_layout.comm();
        let this_process = comm.this_process();

        // We first need to send around the dimension of the array to all processes.
        let mut my_shape = self.shape();

        let other_dims_count = my_shape.iter().skip(1).product::<usize>();

        this_process.broadcast_into(my_shape.as_mut_slice());

        // We now have the shape of the array on all processes. Let's prepare the sending of the
        // data.

        // We evaluate into a new array to ensure that we are row-major contiguous.
        let send_arr = StridedDynArray::row_major_from(self);

        // Take the first comm.size elements of the counts. (counts is 1 + comm.size long)
        // We also have to multiply the counts by the number of elements in the second dimension
        // onwards. This is the same on every process as only the first dimension is distributed.
        let counts = index_layout
            .local_counts()
            .iter()
            .map(|&x| other_dims_count * x)
            .collect::<Vec<usize>>();

        // We can now scatter the data around.

        let my_data = scatterv_root(comm, &counts, send_arr.data());
        // We wrap this data into an array view and then transpose it to get a standard column-major array.
        {
            let mut local_shape = my_shape;
            local_shape[0] = index_layout.number_of_local_indices();
            let local_arr = DynArray::new_from(&StridedSliceArray::from_shape_and_stride(
                &my_data,
                local_shape,
                row_major_stride_from_shape(local_shape),
            ));
            DistributedArray::new(index_layout, local_arr)
        }
    }

    /// Scatter the array out to all nodes using the given `index_layout`.
    ///
    /// The data is always scattered out along the first axis of the array.
    /// Call this method on all ranks that are not `root`.
    pub fn scatter_from_one<'a, C: Communicator>(
        root: usize,
        index_layout: Rc<IndexLayout<'a, C>>,
    ) -> DistributedArray<'a, C, BaseArray<VectorContainer<Item>, NDIM>, NDIM> {
        let comm = index_layout.comm();

        let my_shape = {
            let mut my_shape = [0_usize; NDIM];

            comm.process_at_rank(root as Rank)
                .broadcast_into(my_shape.as_mut_slice());

            my_shape[0] = index_layout.number_of_local_indices();
            my_shape
        };
        // We receive the data from the root process.
        let my_data = scatterv::<Item>(comm, root);

        // The data comes over in row-major order. We create a row-major view and
        // copy it into a new column major array.

        // We wrap this data into an array view and then copy it over to a column major DynArray.
        {
            let local_arr = DynArray::new_from(&StridedSliceArray::from_shape_and_stride(
                &my_data,
                my_shape,
                row_major_stride_from_shape(my_shape),
            ));
            DistributedArray::new(index_layout, local_arr)
        }
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM>,
{
    /// Return the shape of the global array.
    pub fn shape(&self) -> [usize; NDIM] {
        let mut shape = self.local.shape();
        shape[0] = self.index_layout.number_of_global_indices();
        shape
    }

    /// Return the length of the global array.
    ///
    /// For n-dimensional array this is the product of all dimensions.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Return true if the array is empty.
    pub fn is_empty(&self) -> bool {
        *self.shape().iter().min().unwrap() == 0
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> BaseItem for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<'a, C, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: UnsafeRandom1DAccessMut + Shape<NDIM>,
{
    /// Fill from an other distributed array.
    pub fn fill_from<ArrayImplOther>(
        &mut self,
        other: &DistributedArray<'a, C, ArrayImplOther, NDIM>,
    ) where
        ArrayImplOther: Shape<NDIM> + UnsafeRandom1DAccessByValue<Item = ArrayImpl::Item>,
    {
        assert!(
            self.is_compatible_with(other),
            "DistributedArray::fill_from: The index layout and shape of the arrays do not match."
        );
        // We can just fill from the local data.
        self.local.fill_from(&other.local);
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessMut,
    Self: BaseItem<Item = ArrayImpl::Item>,
{
    /// Fill global array with `value`.
    pub fn fill_with_value(&mut self, value: ArrayImpl::Item) {
        self.local.fill_with_value(value);
    }
}

impl<'a, Item, C, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessByValue<Item = Item>,
    Item: std::ops::Add<Output = Item> + Equivalence + Copy + Default,
{
    /// Compute the sum of all elements of the global array.
    pub fn sum(&self) -> Option<Item> {
        if self.is_empty() {
            return None;
        }
        let local_sum = self
            .local
            .iter_value()
            .reduce(std::ops::Add::add)
            .unwrap_or_default();
        let mut global_sum = Default::default();
        self.index_layout.comm().all_reduce_into(
            &local_sum,
            &mut global_sum,
            mpi::collective::SystemOperation::sum(),
        );
        Some(global_sum)
    }
}

impl<'a, C, ArrayImpl, ArrayImplEval, const NDIM: usize> EvaluateObject
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImplEval: Shape<NDIM>,
    Array<ArrayImpl, NDIM>: EvaluateObject<Output = Array<ArrayImplEval, NDIM>>,
{
    type Output = DistributedArray<'a, C, ArrayImplEval, NDIM>;

    fn eval(&self) -> Self::Output {
        DistributedArray::new(self.index_layout.clone(), self.local.eval())
    }
}

impl<'a, C, ArrayImpl, ArrayImplEval, const NDIM: usize> EvaluateRowMajorArray
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImplEval: Shape<NDIM>,
    Array<ArrayImpl, NDIM>: EvaluateRowMajorArray<Output = Array<ArrayImplEval, NDIM>>,
{
    type Output = DistributedArray<'a, C, ArrayImplEval, NDIM>;

    /// Evaluate the global array into a row major global array.
    fn eval_row_major(&self) -> Self::Output {
        DistributedArray::new(self.index_layout.clone(), self.local.eval_row_major())
    }
}

impl<'a, C, Item, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
{
    /// Convert an array into the new item type T.
    ///
    /// Note: It is required that `ArrayImpl::Item: Into<T>`.
    #[allow(clippy::type_complexity)]
    pub fn into_type<T>(
        self,
    ) -> DistributedArray<
        'a,
        C,
        ArrayUnaryOperator<ArrayImpl::Item, T, ArrayImpl, fn(ArrayImpl::Item) -> T, NDIM>,
        NDIM,
    >
    where
        Item: Into<T>,
    {
        DistributedArray::new(self.index_layout.clone(), self.local.into_type())
    }
}

impl<'a, C, Item, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessByValue<Item = Item>,
    Item: Copy + Default + std::ops::Mul<Output = Item>,
{
    /// Componentwise multiply the array with `alpha`.
    pub fn scalar_mul(
        self,
        alpha: Item,
    ) -> DistributedArray<'a, C, ArrayScalarMult<Item, ArrayImpl, NDIM>, NDIM> {
        DistributedArray::new(self.index_layout.clone(), self.local.scalar_mul(alpha))
    }
}

impl<'a, C, ArrayImpl, Item> DistributedArray<'a, C, ArrayImpl, 1>
where
    C: Communicator,
    ArrayImpl: Shape<1> + UnsafeRandom1DAccessByValue<Item = Item>,
    Item: Conj<Output = Item>
        + std::ops::Mul<Output = Item>
        + std::ops::Add<Output = Item>
        + Equivalence
        + Copy
        + Default,
{
    /// Compute the inner product of two distributed 1d arrays.
    ///
    /// Return `None` if the arrays are empty.
    ///
    /// Note: The values of `other` are taken as conjugate.
    pub fn inner<ArrayImplOther>(
        &self,
        other: &DistributedArray<'a, C, ArrayImplOther, 1>,
    ) -> Option<Item>
    where
        ArrayImplOther: Shape<1> + UnsafeRandom1DAccessByValue<Item = Item>,
    {
        assert!(
            self.is_compatible_with(other),
            "DistributedArray::inner: The index layout and shape of the arrays do not match."
        );

        // Return None if the array is empty.
        if self.is_empty() {
            return None;
        }

        // We can just sum from the local data. We know that the array is not empty.
        // But the local array can still be empty, need to deal with this.
        let local_inner = self.local.inner(&other.local).unwrap_or_default();
        let mut global_inner = Default::default();

        self.index_layout.comm().all_reduce_into(
            &local_inner,
            &mut global_inner,
            mpi::collective::SystemOperation::sum(),
        );

        Some(global_inner)
    }
}

impl<'a, C, ArrayImpl, Item, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessByValue,
    ArrayImpl::Item: Abs<Output = Item>,
    Item: Max<Output = Item> + Copy + Default + Equivalence,
{
    /// Compute the maximum absolute value over all elements of the distributed array.
    ///
    /// Return `None` if the array is empty.
    pub fn max_abs(&self) -> Option<Item> {
        if self.is_empty() {
            return None;
        }

        let local_norm_sup = self.local.max_abs().unwrap_or_default();
        let mut global_result = Default::default();
        self.index_layout.comm().all_reduce_into(
            &local_norm_sup,
            &mut global_result,
            mpi::collective::SystemOperation::max(),
        );
        Some(global_result)
    }
}

impl<'a, C, ArrayImpl, Item> DistributedArray<'a, C, ArrayImpl, 1>
where
    C: Communicator,
    ArrayImpl: Shape<1> + UnsafeRandom1DAccessByValue,
    ArrayImpl::Item: Copy + Default + AbsSquare<Output = Item>,
    Item: std::ops::Add<Output = Item> + Sqrt<Output = Item> + Equivalence + Copy + Default,
{
    /// Compute the 2-norm of a distributed vector.
    ///
    /// Return `None` if vector is empty.
    pub fn norm_2(&self) -> Option<Item> {
        self.r().abs_square().sum().map(|elem| elem.sqrt())
    }
}

macro_rules! impl_unary_op_trait {
    ($name:ident, $method_name:ident) => {
        use crate::traits::number_traits::$name;
        impl<'a, C, ArrayImpl, Out, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
        where
            C: Communicator,
            ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessByValue,
            ArrayImpl::Item: $name<Output = Out>,
            Out: Copy + Default + Equivalence,
        {
            #[doc = "Componentwise apply the operation to the distributed array."]
            pub fn $method_name(
                self,
            ) -> DistributedArray<
                'a,
                C,
                ArrayUnaryOperator<
                    ArrayImpl::Item,
                    Out,
                    ArrayImpl,
                    fn(ArrayImpl::Item) -> Out,
                    NDIM,
                >,
                NDIM,
            > {
                DistributedArray::new(self.index_layout.clone(), self.local.$method_name())
            }
        }
    };
}

impl_unary_op_trait!(Abs, abs);
impl_unary_op_trait!(Square, square);
impl_unary_op_trait!(AbsSquare, abs_square);
impl_unary_op_trait!(Sqrt, sqrt);
impl_unary_op_trait!(Exp, exp);
impl_unary_op_trait!(Ln, ln);
impl_unary_op_trait!(Recip, recip);
impl_unary_op_trait!(Sin, sin);
impl_unary_op_trait!(Cos, cos);
impl_unary_op_trait!(Tan, tan);
impl_unary_op_trait!(Asin, asin);
impl_unary_op_trait!(Acos, acos);
impl_unary_op_trait!(Atan, atan);
impl_unary_op_trait!(Sinh, sinh);
impl_unary_op_trait!(Cosh, cosh);
impl_unary_op_trait!(Tanh, tanh);
impl_unary_op_trait!(Asinh, asinh);
impl_unary_op_trait!(Acosh, acosh);
impl_unary_op_trait!(Atanh, atanh);

// Now implement the traits for the standard operators
impl<'a, C, ArrayImpl, ArrayImplOther, const NDIM: usize>
    std::ops::Add<DistributedArray<'a, C, ArrayImplOther, NDIM>>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>: std::ops::Add<
            Array<ArrayImplOther, NDIM>,
            Output = Array<ArrayAddition<ArrayImpl, ArrayImplOther, NDIM>, NDIM>,
        >,
    ArrayImpl: Shape<NDIM>,
    ArrayImplOther: Shape<NDIM>,
{
    type Output = DistributedArray<'a, C, ArrayAddition<ArrayImpl, ArrayImplOther, NDIM>, NDIM>;

    fn add(self, rhs: DistributedArray<'a, C, ArrayImplOther, NDIM>) -> Self::Output {
        assert!(
            self.is_compatible_with(&rhs),
            "DistributedArray::add: The index layout and shape of the arrays do not match."
        );
        DistributedArray::new(self.index_layout.clone(), self.local + rhs.local)
    }
}

impl<'a, C, ArrayImpl, ArrayImplOther, const NDIM: usize>
    std::ops::Sub<DistributedArray<'a, C, ArrayImplOther, NDIM>>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>: std::ops::Sub<
            Array<ArrayImplOther, NDIM>,
            Output = Array<ArraySubtraction<ArrayImpl, ArrayImplOther, NDIM>, NDIM>,
        >,
    ArrayImpl: Shape<NDIM>,
    ArrayImplOther: Shape<NDIM>,
{
    type Output = DistributedArray<'a, C, ArraySubtraction<ArrayImpl, ArrayImplOther, NDIM>, NDIM>;

    fn sub(self, rhs: DistributedArray<'a, C, ArrayImplOther, NDIM>) -> Self::Output {
        assert!(
            self.is_compatible_with(&rhs),
            "DistributedArray::add: The index layout and shape of the arrays do not match."
        );
        DistributedArray::new(self.index_layout.clone(), self.local - rhs.local)
    }
}

impl<'a, C, ArrayImpl, ArrayImplOther, const NDIM: usize>
    std::ops::Mul<DistributedArray<'a, C, ArrayImplOther, NDIM>>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>: std::ops::Mul<
            Array<ArrayImplOther, NDIM>,
            Output = Array<CmpWiseProduct<ArrayImpl, ArrayImplOther, NDIM>, NDIM>,
        >,
    ArrayImpl: Shape<NDIM>,
    ArrayImplOther: Shape<NDIM>,
{
    type Output = DistributedArray<'a, C, CmpWiseProduct<ArrayImpl, ArrayImplOther, NDIM>, NDIM>;

    fn mul(self, rhs: DistributedArray<'a, C, ArrayImplOther, NDIM>) -> Self::Output {
        assert!(
            self.is_compatible_with(&rhs),
            "DistributedArray::add: The index layout and shape of the arrays do not match."
        );
        DistributedArray::new(self.index_layout.clone(), self.local * rhs.local)
    }
}

impl<'a, C, ArrayImpl, ArrayImplOther, const NDIM: usize>
    std::ops::Div<DistributedArray<'a, C, ArrayImplOther, NDIM>>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>: std::ops::Div<
            Array<ArrayImplOther, NDIM>,
            Output = Array<CmpWiseDivision<ArrayImpl, ArrayImplOther, NDIM>, NDIM>,
        >,
    ArrayImpl: Shape<NDIM>,
    ArrayImplOther: Shape<NDIM>,
{
    type Output = DistributedArray<'a, C, CmpWiseDivision<ArrayImpl, ArrayImplOther, NDIM>, NDIM>;

    fn div(self, rhs: DistributedArray<'a, C, ArrayImplOther, NDIM>) -> Self::Output {
        assert!(
            self.is_compatible_with(&rhs),
            "DistributedArray::add: The index layout and shape of the arrays do not match."
        );
        DistributedArray::new(self.index_layout.clone(), self.local / rhs.local)
    }
}

macro_rules! impl_scalar_mult {
    ($scalar:ty) => {
        impl<'a, C, ArrayImpl, const NDIM: usize>
            std::ops::Mul<DistributedArray<'a, C, ArrayImpl, NDIM>> for $scalar
        where
            C: Communicator,
            ArrayImpl: Shape<NDIM> + BaseItem<Item = $scalar>,
            $scalar: std::ops::Mul<
                    Array<ArrayImpl, NDIM>,
                    Output = Array<ArrayScalarMult<$scalar, ArrayImpl, NDIM>, NDIM>,
                >,
        {
            type Output = DistributedArray<'a, C, ArrayScalarMult<$scalar, ArrayImpl, NDIM>, NDIM>;

            fn mul(self, rhs: DistributedArray<'a, C, ArrayImpl, NDIM>) -> Self::Output {
                DistributedArray::new(rhs.index_layout.clone(), self.mul(rhs.local))
            }
        }
    };
}

impl_scalar_mult!(f64);
impl_scalar_mult!(f32);
impl_scalar_mult!(c64);
impl_scalar_mult!(c32);
impl_scalar_mult!(usize);
impl_scalar_mult!(i8);
impl_scalar_mult!(i16);
impl_scalar_mult!(i32);
impl_scalar_mult!(i64);
impl_scalar_mult!(u8);
impl_scalar_mult!(u16);
impl_scalar_mult!(u32);
impl_scalar_mult!(u64);

impl<'a, C, Item, ArrayImpl, const NDIM: usize> std::ops::Mul<Item>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM> + BaseItem<Item = Item>,
    Array<ArrayImpl, NDIM>:
        std::ops::Mul<Item, Output = Array<ArrayScalarMult<Item, ArrayImpl, NDIM>, NDIM>>,
{
    type Output = DistributedArray<'a, C, ArrayScalarMult<Item, ArrayImpl, NDIM>, NDIM>;

    fn mul(self, rhs: Item) -> Self::Output {
        DistributedArray::new(self.index_layout.clone(), self.local.mul(rhs))
    }
}

impl<'a, C, OutImpl, ArrayImpl, const NDIM: usize> std::ops::Neg
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>: std::ops::Neg<Output = Array<OutImpl, NDIM>>,
    OutImpl: Shape<NDIM>,
{
    type Output = DistributedArray<'a, C, OutImpl, NDIM>;

    fn neg(self) -> Self::Output {
        DistributedArray::new(self.index_layout.clone(), self.local.neg())
    }
}

impl<'a, C, ArrayImpl, ArrayImplOther, const NDIM: usize>
    AddAssign<DistributedArray<'a, C, ArrayImplOther, NDIM>>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM>,
    ArrayImplOther: Shape<NDIM>,
    Array<ArrayImpl, NDIM>: AddAssign<Array<ArrayImplOther, NDIM>>,
{
    fn add_assign(&mut self, rhs: DistributedArray<'a, C, ArrayImplOther, NDIM>) {
        assert!(
            self.is_compatible_with(&rhs),
            "DistributedArray::add_assign: The index layout and shape of the arrays do not match."
        );
        self.local += rhs.local;
    }
}

impl<'a, C, ArrayImpl, ArrayImplOther, const NDIM: usize>
    SubAssign<DistributedArray<'a, C, ArrayImplOther, NDIM>>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM>,
    ArrayImplOther: Shape<NDIM>,
    Array<ArrayImpl, NDIM>: SubAssign<Array<ArrayImplOther, NDIM>>,
{
    fn sub_assign(&mut self, rhs: DistributedArray<'a, C, ArrayImplOther, NDIM>) {
        assert!(
            self.is_compatible_with(&rhs),
            "DistributedArray::sub_assign: The index layout and shape of the arrays do not match."
        );
        self.local -= rhs.local;
    }
}

impl<'a, C, Other, ArrayImpl, const NDIM: usize> MulAssign<Other>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>: MulAssign<Other>,
{
    fn mul_assign(&mut self, rhs: Other) {
        self.local *= rhs;
    }
}

impl<'a, C, Other, ArrayImpl, const NDIM: usize> DivAssign<Other>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>: DivAssign<Other>,
{
    fn div_assign(&mut self, rhs: Other) {
        self.local /= rhs;
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM>,
{
    /// Cast the distributed array to a different type.
    pub fn cast<Target>(self) -> DistributedArray<'a, C, ArrayCast<Target, ArrayImpl, NDIM>, NDIM> {
        DistributedArray::new(self.index_layout.clone(), self.local.cast::<Target>())
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM>,
{
    /// Cast the distributed array to a different dimensionality.
    pub fn coerce_dim<const CDIM: usize>(
        self,
    ) -> RlstResult<DistributedArray<'a, C, CoerceArray<ArrayImpl, NDIM, CDIM>, CDIM>> {
        Ok(DistributedArray::new(
            self.index_layout.clone(),
            self.local.coerce_dim::<CDIM>()?,
        ))
    }
}

impl<'a, C, Item, ArrayImpl1, ArrayImpl2, const NDIM: usize>
    MulAdd<Item, DistributedArray<'a, C, ArrayImpl2, NDIM>>
    for DistributedArray<'a, C, ArrayImpl1, NDIM>
where
    C: Communicator,
    ArrayImpl1: Shape<NDIM>,
    ArrayImpl2: Shape<NDIM>,
    Array<ArrayImpl1, NDIM>: MulAdd<
            Item,
            Array<ArrayImpl2, NDIM>,
            Output = Array<MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>, NDIM>,
        >,
{
    type Output = DistributedArray<'a, C, MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>, NDIM>;

    fn mul_add(self, alpha: Item, rhs: DistributedArray<'a, C, ArrayImpl2, NDIM>) -> Self::Output {
        assert!(
            self.is_compatible_with(&rhs),
            "DistributedArray::mul_add: The index layout and shape of the arrays do not match."
        );
        DistributedArray::new(
            self.index_layout.clone(),
            self.local.mul_add(alpha, rhs.local),
        )
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM>,
{
    /// Apply a unary operator to the distributed array.
    pub fn unary_op<OpItem, OpTarget, Op: Fn(OpItem) -> OpTarget>(
        self,
        op: Op,
    ) -> DistributedArray<'a, C, ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>, NDIM>
    {
        DistributedArray::new(self.index_layout.clone(), self.local.unary_op(op))
    }
}

impl<C: Communicator, ArrayImpl> DistributedArray<'_, C, ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessByValue<2>,
{
    /// Return a column iterator for a distributed 2d array.
    pub fn col_iter(&self) -> DistributedColIterator<'_, C, ArrayImpl> {
        DistributedColIterator::new(self)
    }
}

impl<'it, 'a, C: Communicator, ArrayImpl> DistributedArray<'a, C, ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessMut<2>,
{
    /// Return a mutable column iterator for a distributed 2d array.
    pub fn col_iter_mut(&'it mut self) -> DistributedColIteratorMut<'it, 'a, C, ArrayImpl, 2> {
        DistributedColIteratorMut::new(self)
    }
}

/// Distributed Column iterator
pub struct DistributedColIterator<'a, C, ArrayImpl>
where
    C: Communicator,
{
    arr: &'a DistributedArray<'a, C, ArrayImpl, 2>,
    ncols: usize,
    current_col: usize,
}

impl<'a, C, ArrayImpl> DistributedColIterator<'a, C, ArrayImpl>
where
    C: Communicator,
    ArrayImpl: Shape<2>,
{
    /// Create a new column iterator for the given array.
    pub fn new(arr: &'a DistributedArray<'a, C, ArrayImpl, 2>) -> Self {
        let ncols = arr.shape()[1];
        DistributedColIterator {
            arr,
            ncols,
            current_col: 0,
        }
    }
}

/// Mutable distributed column iterator
pub struct DistributedColIteratorMut<'it, 'a, C, ArrayImpl, const NDIM: usize>
where
    C: Communicator,
    'a: 'it,
{
    arr: &'it mut DistributedArray<'a, C, ArrayImpl, NDIM>,
    ncols: usize,
    current_col: usize,
}

impl<'it, 'a, C, ArrayImpl> DistributedColIteratorMut<'it, 'a, C, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
    C: Communicator,
    'a: 'it,
{
    /// Create a new column iterator for the given array.
    pub fn new(arr: &'it mut DistributedArray<'a, C, ArrayImpl, 2>) -> Self {
        let ncols = arr.shape()[1];
        DistributedColIteratorMut {
            arr,
            ncols,
            current_col: 0,
        }
    }
}

impl<'a, C: Communicator, ArrayImpl: Shape<2>> std::iter::Iterator
    for DistributedColIterator<'a, C, ArrayImpl>
{
    type Item = DistributedArray<'a, C, ArraySlice<reference::ArrayRef<'a, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.ncols {
            return None;
        }
        let slice = DistributedArray::new(
            self.arr.index_layout.clone(),
            self.arr.local.r().slice(1, self.current_col),
        );
        self.current_col += 1;
        Some(slice)
    }
}

impl<'it, 'a, C: Communicator, ArrayImpl> std::iter::Iterator
    for DistributedColIteratorMut<'it, 'a, C, ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + 'a,
    'a: 'it,
{
    type Item = DistributedArray<'a, C, ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.ncols {
            return None;
        }
        let slice = DistributedArray::new(
            self.arr.index_layout.clone(),
            self.arr.local.r_mut().slice(1, self.current_col),
        );
        self.current_col += 1;
        unsafe {
            Some(std::mem::transmute::<
                DistributedArray<'_, C, ArraySlice<ArrayRefMut<'_, ArrayImpl, 2>, 2, 1>, 1>,
                DistributedArray<'a, C, ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>,
            >(slice))
        }
    }
}

/// Create a new distributed vector with the given scalar type and index layout.
///
/// The index layout must be provided as `RefCell`, that is as [Rc<IndexLayout>].
#[macro_export]
macro_rules! dist_vec {
    ($scalar:ty, $index_layout:expr) => {
        $crate::sparse::distributed_array::DistributedArray::new(
            $index_layout,
            $crate::dense::array::DynArray::<$scalar, 1>::from_shape([
                $index_layout.number_of_local_indices()
            ]),
        )
    };
}

/// Create a new distributed array.
///
/// The number of rows is determined by the index layout, while the number of columns is
/// determined by `ncols`.
///
/// The index layout must be provided as `RefCell`, that is as [Rc<IndexLayout>].
#[macro_export]
macro_rules! dist_mat {
    ($scalar:ty, $index_layout:expr, ncols:expr) => {
        $crate::sparse::distributed_array::DistributedArray::new(
            $index_layout,
            $crate::DynArray::<$scalar, 2>::new([$index_layout.number_of_local_indices(), ncols]),
        )
    };
}
