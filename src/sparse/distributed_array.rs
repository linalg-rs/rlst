//! An Indexable Vector is a container whose elements can be 1d indexed.
use std::rc::Rc;

use crate::dense::array::operators::unary_op::ArrayUnaryOperator;
use crate::dense::array::reference::{ArrayRef, ArrayRefMut};
use crate::dense::base_array::BaseArray;
use crate::dense::data_container::VectorContainer;
use crate::dense::layout::row_major_stride_from_shape;
use crate::distributed_tools::{scatterv, scatterv_root, IndexLayout};
use paste::paste;

use crate::dense::array::{DynArray, StridedDynArray, StridedSliceArray};
use crate::{
    Array, BaseItem, CmpMulAddFrom, CmpMulFrom, ConjArray, EvaluateArray, FillFrom, FillFromResize,
    FillWithValue, GatherToOne, Inner, Len, NormSup, NormTwo, NumberOfElements, RawAccess,
    RawAccessMut, ScaleInPlace, ScatterFromOne, Shape, Sqrt, Sum, SumFrom, ToType,
};
use crate::{EvaluateRowMajorArray, GatherToAll};

use mpi::datatype::PartitionMut;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence, Root};
use mpi::Rank;

/// Distributed Array.
///
/// A distributed array is a an array that is distributed along the first dimension with
/// respect to a given index layout.
pub struct DistributedArray<'a, C: Communicator, ArrayImpl, const NDIM: usize> {
    /// The index layout of the vector.
    pub index_layout: Rc<IndexLayout<'a, C>>,
    /// The local data of the vector
    pub local: Array<ArrayImpl, NDIM>, // A RefCell is necessary as we often need a reference to the communicator and mutable ref to local at the same time.
                                       // But this would be disallowed by Rust's static borrow checker.
}

impl<'a, C, ArrayImpl, const NDIM: usize> DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM>,
{
    /// Crate new
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
    fn is_compatible_with<ArrayImplOther>(
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

impl<'a, C, ArrayImpl, const NDIM: usize> GatherToAll for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM> + BaseItem,
    Array<ArrayImpl, NDIM>: EvaluateRowMajorArray,
    <Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output: RawAccess,
    <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item:
        Equivalence + Clone + Default,
    StridedDynArray<
        <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item,
        NDIM,
    >: EvaluateArray,
{
    type Output = <StridedDynArray<
        <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item,
        NDIM,
    > as EvaluateArray>::Output;
    /// Gather `Self` to all processes and store in `arr`.
    fn gather_to_all(&self) -> Self::Output {
        let comm = self.index_layout.comm();
        let this_process = comm.this_process();

        // We evaluate into a new array to ensure that we are row-major contiguous.
        let send_arr = self.local.eval_row_major();

        // The local shape of the array.
        let local_shape = self.local.shape();
        // This is the number of elements from the second dimension onwards.
        // This is the same on every process as only the first dmension is distributed.
        // Since an empty iterator returns a one value this line works also if we have a 1D array.
        let other_dims_count = local_shape.iter().skip(1).product::<usize>();
        // This is going to be the new global shape. All dimensions except the first one are the
        // same as the local shape. The first dimension is the number of global indices.
        let global_shape = {
            let mut tmp = local_shape.clone();
            tmp[0] = self.index_layout.number_of_global_indices();
            tmp
        };
        // We create a new row-major array to send data into.
        // This is necessary as we are sending row slices
        let mut recv_arr = StridedDynArray::<
            <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item,
            NDIM,
        >::row_major(global_shape);

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

impl<'a, C, ArrayImpl, const NDIM: usize> GatherToOne for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM> + BaseItem,
    Array<ArrayImpl, NDIM>: EvaluateRowMajorArray,
    <Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output: RawAccess,
    <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item:
        Equivalence + Clone + Default,
    StridedDynArray<
        <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item,
        NDIM,
    >: EvaluateArray,
{
    type Output = <StridedDynArray<
        <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item,
        NDIM,
    > as EvaluateArray>::Output;

    fn gather_to_one(&self, root: usize) {
        let comm = self.index_layout.comm();
        let send_arr = self.local.eval_row_major();
        let target_process = comm.process_at_rank(root as Rank);

        target_process.gather_varcount_into(send_arr.data());
    }

    fn gather_to_one_root(&self) -> Self::Output {
        let comm = self.index_layout.comm();
        let this_process = comm.this_process();

        // We evaluate into a new array to ensure that we are row-major contiguous.
        let send_arr = self.local.eval_row_major();

        // The local shape of the array.
        let local_shape = self.local.shape();
        // This is the number of elements from the second dimension onwards.
        // This is the same on every process as only the first dmension is distributed.
        // Since an empty iterator returns a one value this line works also if we have a 1D array.
        let other_dims_count = local_shape.iter().skip(1).product::<usize>();
        // This is going to be the new global shape. All dimensions except the first one are the
        // same as the local shape. The first dimension is the number of global indices.
        let global_shape = {
            let mut tmp = local_shape.clone();
            tmp[0] = self.index_layout.number_of_global_indices();
            tmp
        };
        // We create a new row-major array to send data into.
        // This is necessary as we are sending row slices
        let mut recv_arr = StridedDynArray::<
            <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item,
            NDIM,
        >::row_major(global_shape);

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

impl<ArrayImpl, const NDIM: usize> ScatterFromOne for Array<ArrayImpl, NDIM>
where
    Array<ArrayImpl, NDIM>: EvaluateRowMajorArray + Shape<NDIM>,
    <Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output: RawAccess,
    <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item:
        Equivalence + Copy + Default,
    for<'b> DynArray<<<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item, NDIM>:
        FillFromResize<
            StridedSliceArray<
                'b,
                <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item,
                NDIM,
            >,
        >,
{
    type Output<'a, C>
        = DistributedArray<
        'a,
        C,
        BaseArray<
            VectorContainer<
                <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item,
            >,
            NDIM,
        >,
        NDIM,
    >
    where
        C: 'a,
        C: Communicator;

    fn scatter_from_one_root<'a, C: Communicator>(
        &self,
        index_layout: Rc<IndexLayout<'a, C>>,
    ) -> Self::Output<'a, C> {
        let comm = index_layout.comm();
        let this_process = comm.this_process();

        // We first need to send around the dimension of the array to all processes.
        let mut my_shape = self.shape();

        let other_dims_count = my_shape.iter().skip(1).product::<usize>();

        this_process.broadcast_into(my_shape.as_mut_slice());

        // We now have the shape of the array on all processes. Let's prepare the sending of the
        // data.

        // We evaluate into a new array to ensure that we are row-major contiguous.
        let send_arr = self.eval_row_major();

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
            let mut local_shape = my_shape.clone();
            local_shape[0] = index_layout.number_of_local_indices();
            let local_arr = DynArray::new_from(&StridedSliceArray::from_shape_and_stride(
                &my_data,
                local_shape,
                row_major_stride_from_shape(local_shape),
            ));
            DistributedArray::new(index_layout, local_arr)
        }
    }

    fn scatter_from_one<'a, C: Communicator>(
        root: usize,
        index_layout: Rc<IndexLayout<'a, C>>,
    ) -> Self::Output<'a, C> {
        let comm = index_layout.comm();

        let my_shape = {
            let mut my_shape = [0_usize; NDIM];

            comm.process_at_rank(root as Rank)
                .broadcast_into(my_shape.as_mut_slice());

            my_shape[0] = index_layout.number_of_local_indices();
            my_shape
        };
        // We receive the data from the root process.
        let my_data = scatterv::<
            <<Array<ArrayImpl, NDIM> as EvaluateRowMajorArray>::Output as BaseItem>::Item,
        >(comm, root);

        // The data comes over in row-major order. We create a row-major view and
        // copy it into a new column major array.

        // We wrap this data into an array view and then transpose it to get a standard column-major array.
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

impl<'a, C, ArrayImpl, const NDIM: usize> Shape<NDIM> for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Shape<NDIM>,
{
    fn shape(&self) -> [usize; NDIM] {
        let mut shape = self.local.shape();
        shape[0] = self.index_layout.number_of_global_indices();
        shape
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> BaseItem for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<'a, C, ArrayImpl, ArrayImplOther, const NDIM: usize>
    FillFrom<DistributedArray<'a, C, ArrayImplOther, NDIM>>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>: FillFrom<Array<ArrayImplOther, NDIM>>,
    ArrayImpl: Shape<NDIM>,
    ArrayImplOther: Shape<NDIM>,
{
    fn fill_from(&mut self, other: &DistributedArray<'a, C, ArrayImplOther, NDIM>) {
        assert!(
            self.is_compatible_with(other),
            "DistributedArray::fill_from: The index layout and shape of the arrays do not match."
        );
        // We can just fill from the local data.
        self.local.fill_from(&other.local);
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> NumberOfElements
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: NumberOfElements,
    ArrayImpl: Shape<NDIM>,
{
    fn number_of_elements(&self) -> usize {
        self.index_layout.number_of_global_indices()
            * self.local.shape().iter().skip(1).product::<usize>()
    }
}

impl<'a, C, ArrayImpl, ArrayImplOther, const NDIM: usize>
    SumFrom<DistributedArray<'a, C, ArrayImplOther, NDIM>>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>: SumFrom<Array<ArrayImplOther, NDIM>>,
    ArrayImpl: Shape<NDIM>,
    ArrayImplOther: Shape<NDIM>,
{
    fn sum_from(&mut self, other: &DistributedArray<'a, C, ArrayImplOther, NDIM>) {
        assert!(
            self.is_compatible_with(other),
            "DistributedArray::sum_from: The index layout and shape of the arrays do not match."
        );
        // We can just sum from the local data.
        self.local.sum_from(&other.local);
    }
}

impl<'a, C, ArrayImpl, ArrayImplOther, const NDIM: usize>
    CmpMulFrom<DistributedArray<'a, C, ArrayImplOther, NDIM>>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>: CmpMulFrom<Array<ArrayImplOther, NDIM>>,
    ArrayImpl: Shape<NDIM>,
    ArrayImplOther: Shape<NDIM>,
{
    fn cmp_mul_from(&mut self, other: &DistributedArray<'a, C, ArrayImplOther, NDIM>) {
        assert!(
            self.is_compatible_with(other),
            "DistributedArray::sum_from: The index layout and shape of the arrays do not match."
        );
        // We can just sum from the local data.
        self.local.cmp_mul_from(&other.local);
    }
}

impl<'a, C, ArrayImpl, ArrayImplOther1, ArrayImplOther2, const NDIM: usize>
    CmpMulAddFrom<
        DistributedArray<'a, C, ArrayImplOther1, NDIM>,
        DistributedArray<'a, C, ArrayImplOther2, NDIM>,
    > for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>:
        CmpMulAddFrom<Array<ArrayImplOther1, NDIM>, Array<ArrayImplOther2, NDIM>>,
    ArrayImpl: Shape<NDIM>,
    ArrayImplOther1: Shape<NDIM>,
    ArrayImplOther2: Shape<NDIM>,
{
    fn cmp_mul_add_from(
        &mut self,
        other1: &DistributedArray<'a, C, ArrayImplOther1, NDIM>,
        other2: &DistributedArray<'a, C, ArrayImplOther2, NDIM>,
    ) {
        assert!(
            self.is_compatible_with(other1) && self.is_compatible_with(other2),
            "DistributedArray::sum_from: The index layout and shape of the arrays do not match."
        );
        // We can just sum from the local data.
        self.local.cmp_mul_add_from(&other1.local, &other2.local);
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> FillWithValue for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Array<ArrayImpl, NDIM>: FillWithValue,
    Self: BaseItem<Item = <Array<ArrayImpl, NDIM> as BaseItem>::Item>,
{
    fn fill_with_value(&mut self, value: Self::Item) {
        self.local.fill_with_value(value);
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> ScaleInPlace for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Self: BaseItem,
    Array<ArrayImpl, NDIM>: ScaleInPlace<Item = <Self as BaseItem>::Item>,
{
    fn scale_in_place(&mut self, alpha: Self::Item) {
        self.local.scale_in_place(alpha);
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> Sum for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    Self: BaseItem,
    Self::Item: Equivalence + Default,
    Array<ArrayImpl, NDIM>: Sum<Item = <Self as BaseItem>::Item>,
{
    fn sum(&self) -> Self::Item {
        let local_sum = self.local.sum();
        let mut global_sum = Default::default();
        self.index_layout.comm().all_reduce_into(
            &local_sum,
            &mut global_sum,
            mpi::collective::SystemOperation::sum(),
        );
        global_sum
    }
}

impl<'a, C, ArrayImpl, const NDIM: usize> Len for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: Len,
    ArrayImpl: Shape<NDIM>,
{
    fn len(&self) -> usize {
        self.index_layout.number_of_global_indices()
            * self.local.shape().iter().skip(1).product::<usize>()
    }
}

impl<'a, C, ArrayImpl, ArrayImplConj, const NDIM: usize> ConjArray
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImplConj: Shape<NDIM>,
    Array<ArrayImpl, NDIM>: ConjArray<Output = Array<ArrayImplConj, NDIM>>,
{
    type Output = DistributedArray<'a, C, ArrayImplConj, NDIM>;

    fn conj(self) -> Self::Output {
        DistributedArray::new(self.index_layout.clone(), self.local.conj())
    }
}

impl<'a, C, ArrayImpl, ArrayImplEval, const NDIM: usize> EvaluateArray
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImplEval: Shape<NDIM>,
    Array<ArrayImpl, NDIM>: EvaluateArray<Output = Array<ArrayImplEval, NDIM>>,
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

    fn eval_row_major(&self) -> Self::Output {
        DistributedArray::new(self.index_layout.clone(), self.local.eval_row_major())
    }
}

impl<'a, C, T, Item, ArrayImpl, const NDIM: usize> ToType<T>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    ArrayImpl: BaseItem<Item = Item> + Shape<NDIM>,
    Array<ArrayImpl, NDIM>: ToType<
        T,
        Item = Item,
        Output = Array<ArrayUnaryOperator<Item, T, ArrayImpl, fn(Item) -> T, NDIM>, NDIM>,
    >,
{
    type Item = Item;

    type Output =
        DistributedArray<'a, C, ArrayUnaryOperator<Item, T, ArrayImpl, fn(Item) -> T, NDIM>, NDIM>;

    fn into_type(self) -> Self::Output
    where
        Self::Item: Into<T>,
    {
        DistributedArray::new(self.index_layout.clone(), self.local.into_type())
    }
}

impl<'a, C, ArrayImpl, ArrayImplOther, T, const NDIM: usize>
    Inner<DistributedArray<'a, C, ArrayImplOther, NDIM>>
    for DistributedArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
    T: Equivalence + Default,
    ArrayImpl: Shape<NDIM>,
    ArrayImplOther: Shape<NDIM>,
    Array<ArrayImpl, NDIM>: Inner<Array<ArrayImplOther, NDIM>, Output = T>,
{
    type Output = T;

    fn inner(&self, other: &DistributedArray<'a, C, ArrayImplOther, NDIM>) -> Self::Output {
        assert!(
            self.is_compatible_with(other),
            "DistributedArray::inner: The index layout and shape of the arrays do not match."
        );
        // We can just sum from the local data.
        let local_inner = self.local.inner(&other.local);
        let mut global_inner = Default::default();

        self.index_layout.comm().all_reduce_into(
            &local_inner,
            &mut global_inner,
            mpi::collective::SystemOperation::sum(),
        );

        global_inner
    }
}

impl<'a, C, ArrayImpl, T> NormSup for DistributedArray<'a, C, ArrayImpl, 1>
where
    C: Communicator,
    T: Equivalence + Default,
    ArrayImpl: Shape<1>,
    Array<ArrayImpl, 1>: NormSup<Output = T>,
{
    type Output = T;

    fn norm_sup(&self) -> Self::Output {
        let local_norm_sup = self.local.norm_sup();
        let mut global_result = Default::default();
        self.index_layout.comm().all_reduce_into(
            &local_norm_sup,
            &mut global_result,
            mpi::collective::SystemOperation::max(),
        );
        global_result
    }
}

impl<'a, C, ArrayImpl, T> NormTwo for DistributedArray<'a, C, ArrayImpl, 1>
where
    C: Communicator,
    ArrayImpl: Shape<1>,
    for<'b> DistributedArray<'a, C, ArrayRef<'b, ArrayImpl, 1>, 1>: ArrayOpAbsSquare,
    for<'b> <DistributedArray<'a, C, ArrayRef<'b, ArrayImpl, 1>, 1> as ArrayOpAbsSquare>::Output:
        Sum + BaseItem<Item = T>,
    T: Sqrt,
{
    type Output = <T as Sqrt>::Output;

    fn norm_2(&self) -> Self::Output {
        Sqrt::sqrt(&self.r().abs_square().sum())
    }
}

macro_rules! impl_unary_op_trait {
    ($trait_name:expr, $method_name:ident) => {
        paste! {
        use $crate::traits::array::[<ArrayOp $trait_name>];
        impl<'a, C, ArrayImpl, ArrayImplOutput, const NDIM: usize> $crate::traits::array::[<ArrayOp $trait_name>]
            for DistributedArray<'a, C, ArrayImpl, NDIM>
        where
            C: Communicator,
            ArrayImpl: Shape<NDIM>,
            Array<ArrayImpl, NDIM>: [<ArrayOp $trait_name>]<Output = Array<ArrayImplOutput, NDIM>>,
            ArrayImplOutput: Shape<NDIM>,
        {
            type Output = DistributedArray<'a, C, ArrayImplOutput, NDIM>;

            fn $method_name(self) -> Self::Output {
                DistributedArray::new(self.index_layout.clone(), self.local.$method_name())
            }
        }}
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
