//! An Indexable Vector is a container whose elements can be 1d indexed.
use std::rc::Rc;

use crate::dense::base_array::BaseArray;
use crate::dense::data_container::VectorContainer;
use crate::dense::layout::row_major_stride_from_shape;
use crate::distributed_tools::{scatterv, scatterv_root, IndexLayout};

use crate::dense::array::{DynArray, StridedDynArray, StridedSliceArray};
use crate::{
    empty_array, AbsSquare, Array, BaseItem, EvaluateArray, FillFromIter, FillFromResize,
    GatherToOne, Inner, NormSup, NormTwo, RawAccess, RawAccessMut, ScatterFromOne, Shape, Sqrt,
    Stride,
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
{
    /// Crate new
    pub fn new(index_layout: Rc<IndexLayout<'a, C>>, arr: Array<ArrayImpl, NDIM>) -> Self {
        let number_of_local_indices = index_layout.number_of_local_indices();
        DistributedArray {
            index_layout,
            local: arr,
        }
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
            .counts()
            .iter()
            .take(comm.size() as usize)
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
            .counts()
            .iter()
            .take(comm.size() as usize)
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
        Equivalence + Clone + Default,
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
            .counts()
            .iter()
            .take(comm.size() as usize)
            .map(|&x| (other_dims_count * x))
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

    fn scatter_from_one(&self, root: usize) -> Self::Output {
        todo!()
    }
}

//     /// Send vector to a given root rank (call this from the root)
//     pub fn gather_to_rank_root(&self, arr: &mut Array<ArrayImpl, 1>)
//     where
//         ArrayImpl: Shape<1> + RawAccessMut<Item = Item>,
//     {
//         let comm = self.index_layout.comm();
//         let my_process = comm.this_process();
//
//         let global_dim = self.index_layout.number_of_global_indices();
//
//         assert_eq!(arr.shape()[0], global_dim);
//
//         // Take the first comm.size elements of the counts. (counts is 1 + comm.size long)
//         let counts = self
//             .index_layout
//             .counts()
//             .iter()
//             .take(comm.size() as usize)
//             .map(|&x| x as i32)
//             .collect::<Vec<i32>>();
//
//         let displacements: Vec<i32> = crate::distributed_tools::displacements(&counts);
//         let mut partition = PartitionMut::new(arr.data_mut(), counts, displacements);
//
//         my_process.gather_varcount_into_root(self.local.data(), &mut partition);
//     }
//
//     /// Send vector to a given rank (call this from the root)
//     pub fn gather_to_rank(&self, target_rank: usize) {
//         let target_process = self
//             .index_layout
//             .comm()
//             .process_at_rank(target_rank as Rank);
//
//         target_process.gather_varcount_into(self.local.data());
//     }
//
//     /// Create from root
//     pub fn scatter_from_root<ArrayImpl: Shape<1> + RawAccess<Item = Item>>(
//         &mut self,
//         arr: &mut Array<ArrayImpl, 1>,
//     ) where
//         Item: Copy,
//     {
//         let comm = self.index_layout.comm();
//
//         assert_eq!(arr.shape()[0], self.index_layout.number_of_global_indices());
//         // Take the first comm.size elements of the counts. (counts is 1 + comm.size long)
//         let counts = self
//             .index_layout
//             .counts()
//             .iter()
//             .take(comm.size() as usize)
//             .copied()
//             .collect::<Vec<usize>>();
//
//         self.local
//             .fill_from_iter(scatterv_root(comm, &counts, arr.data()).iter().copied());
//     }
//
//     /// Create from root
//     pub fn scatter_from(&mut self, root: usize)
//     where
//         Item: Copy,
//     {
//         let comm = self.index_layout.comm();
//
//         self.local
//             .fill_from_iter(scatterv(comm, root).iter().copied());
//     }
// }
//
// impl<'a, C, Item> Inner<DistributedArray<'a, C, Item>> for DistributedArray<'a, C, Item>
// where
//     C: Communicator,
//     Item: Equivalence + Default,
//     DynArray<Item, 1>: Inner<DynArray<Item, 1>, Output = Item>,
// {
//     type Output = Item;
//
//     fn inner(&self, other: &DistributedArray<'a, C, Item>) -> Self::Output {
//         assert_eq!(
//             self.index_layout.number_of_local_indices(),
//             other.index_layout.number_of_local_indices()
//         );
//
//         let local_inner = self.local.inner(&other.local);
//         let comm = self.index_layout.comm();
//
//         let mut global_result = Default::default();
//         comm.all_reduce_into(
//             &local_inner,
//             &mut global_result,
//             mpi::collective::SystemOperation::sum(),
//         );
//         global_result
//     }
// }
//
// impl<'a, C, Item> AbsSquare for DistributedArray<'a, C, Item>
// where
//     C: Communicator,
//     DynArray<Item, 1>: AbsSquare,
//     <DynArray<Item, 1> as AbsSquare>::Output: Equivalence + Default,
// {
//     type Output = <DynArray<Item, 1> as AbsSquare>::Output;
//
//     fn abs_square(&self) -> Self::Output {
//         assert_eq!(
//             self.index_layout.number_of_local_indices(),
//             self.local.shape()[0]
//         );
//
//         let local_abs_square = self.local.abs_square();
//         let comm = self.index_layout.comm();
//
//         let mut global_result = Default::default();
//         comm.all_reduce_into(
//             &local_abs_square,
//             &mut global_result,
//             mpi::collective::SystemOperation::sum(),
//         );
//         global_result
//     }
// }
//
// impl<'a, C, Item> NormSup for DistributedArray<'a, C, Item>
// where
//     C: Communicator,
//     DynArray<Item, 1>: NormSup,
//     <DynArray<Item, 1> as NormSup>::Output: Equivalence + Default,
// {
//     type Output = <DynArray<Item, 1> as NormSup>::Output;
//
//     fn norm_sup(&self) -> Self::Output {
//         assert_eq!(
//             self.index_layout.number_of_local_indices(),
//             self.local.shape()[0]
//         );
//
//         let local_norm_sup = self.local.norm_sup();
//         let comm = self.index_layout.comm();
//
//         let mut global_result = Default::default();
//         comm.all_reduce_into(
//             &local_norm_sup,
//             &mut global_result,
//             mpi::collective::SystemOperation::max(),
//         );
//         global_result
//     }
// }
//
// impl<'a, C, Item> NormTwo for DistributedArray<'a, C, Item>
// where
//     C: Communicator,
//     Self: AbsSquare,
//     <Self as AbsSquare>::Output: Sqrt,
// {
//     type Output = <<Self as AbsSquare>::Output as Sqrt>::Output;
//
//     fn norm_2(&self) -> Self::Output {
//         Sqrt::sqrt(&self.abs_square())
//     }
// }
