//! An Indexable Vector is a container whose elements can be 1d indexed.
use std::rc::Rc;

use crate::distributed_tools::{scatterv, scatterv_root, IndexLayout};

use crate::dense::array::DynArray;
use crate::{
    AbsSquare, Array, BaseItem, FillFromIter, Inner, NormSup, NormTwo, RawAccess, RawAccessMut,
    Shape, Sqrt,
};
use mpi::datatype::PartitionMut;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence, Root};
use mpi::Rank;

/// Distributed vector
pub struct DistributedVector<'a, C: Communicator, Item> {
    /// The index layout of the vector.
    pub index_layout: Rc<IndexLayout<'a, C>>,
    /// The local data of the vector
    pub local: DynArray<Item, 1>, // A RefCell is necessary as we often need a reference to the communicator and mutable ref to local at the same time.
                                  // But this would be disallowed by Rust's static borrow checker.
}

impl<'a, C, Item> DistributedVector<'a, C, Item>
where
    C: Communicator,
    Item: Equivalence,
{
    /// Crate new
    pub fn new(index_layout: Rc<IndexLayout<'a, C>>) -> Self
    where
        Item: Copy + Default,
    {
        let number_of_local_indices = index_layout.number_of_local_indices();
        DistributedVector {
            index_layout,
            local: DynArray::from_shape([number_of_local_indices]),
        }
    }

    /// Gather `Self` to all processes and store in `arr`.
    pub fn gather_to_all<ArrayImpl>(&self, arr: &mut Array<ArrayImpl, 1>)
    where
        ArrayImpl: BaseItem<Item = Item> + Shape<1> + RawAccessMut,
    {
        let comm = self.index_layout.comm();
        assert_eq!(self.index_layout.number_of_global_indices(), arr.shape()[0]);

        let this_process = comm.this_process();

        // Take the first comm.size elements of the counts. (counts is 1 + comm.size long)
        let counts = self
            .index_layout
            .counts()
            .iter()
            .take(comm.size() as usize)
            .map(|&x| x as i32)
            .collect::<Vec<i32>>();

        let displacements: Vec<i32> = crate::distributed_tools::displacements(&counts);
        let mut partition = PartitionMut::new(arr.data_mut(), counts, displacements);
        this_process.all_gather_varcount_into(self.local.data(), &mut partition);
    }

    /// Send vector to a given rank (call this from the root)
    pub fn gather_to_rank_root<ArrayImpl>(&self, arr: &mut Array<ArrayImpl, 1>)
    where
        ArrayImpl: Shape<1> + RawAccessMut<Item = Item>,
    {
        let comm = self.index_layout.comm();
        let my_process = comm.this_process();

        let global_dim = self.index_layout.number_of_global_indices();

        assert_eq!(arr.shape()[0], global_dim);

        // Take the first comm.size elements of the counts. (counts is 1 + comm.size long)
        let counts = self
            .index_layout
            .counts()
            .iter()
            .take(comm.size() as usize)
            .map(|&x| x as i32)
            .collect::<Vec<i32>>();

        let displacements: Vec<i32> = crate::distributed_tools::displacements(&counts);
        let mut partition = PartitionMut::new(arr.data_mut(), counts, displacements);

        my_process.gather_varcount_into_root(self.local.data(), &mut partition);
    }

    /// Send vector to a given rank (call this from the root)
    pub fn gather_to_rank(&self, target_rank: usize) {
        let target_process = self
            .index_layout
            .comm()
            .process_at_rank(target_rank as Rank);

        target_process.gather_varcount_into(self.local.data());
    }

    /// Create from root
    pub fn scatter_from_root<ArrayImpl: Shape<1> + RawAccess<Item = Item>>(
        &mut self,
        arr: &mut Array<ArrayImpl, 1>,
    ) where
        Item: Copy,
    {
        let comm = self.index_layout.comm();

        assert_eq!(arr.shape()[0], self.index_layout.number_of_global_indices());
        // Take the first comm.size elements of the counts. (counts is 1 + comm.size long)
        let counts = self
            .index_layout
            .counts()
            .iter()
            .take(comm.size() as usize)
            .copied()
            .collect::<Vec<usize>>();

        self.local
            .fill_from_iter(scatterv_root(comm, &counts, arr.data()).iter().copied());
    }

    /// Create from root
    pub fn scatter_from(&mut self, root: usize)
    where
        Item: Copy,
    {
        let comm = self.index_layout.comm();

        self.local
            .fill_from_iter(scatterv(comm, root).iter().copied());
    }
}

impl<'a, C, Item> Inner<DistributedVector<'a, C, Item>> for DistributedVector<'a, C, Item>
where
    C: Communicator,
    Item: Equivalence + Default,
    DynArray<Item, 1>: Inner<DynArray<Item, 1>, Output = Item>,
{
    type Output = Item;

    fn inner(&self, other: &DistributedVector<'a, C, Item>) -> Self::Output {
        assert_eq!(
            self.index_layout.number_of_local_indices(),
            other.index_layout.number_of_local_indices()
        );

        let local_inner = self.local.inner(&other.local);
        let comm = self.index_layout.comm();

        let mut global_result = Default::default();
        comm.all_reduce_into(
            &local_inner,
            &mut global_result,
            mpi::collective::SystemOperation::sum(),
        );
        global_result
    }
}

impl<'a, C, Item> AbsSquare for DistributedVector<'a, C, Item>
where
    C: Communicator,
    DynArray<Item, 1>: AbsSquare,
    <DynArray<Item, 1> as AbsSquare>::Output: Equivalence + Default,
{
    type Output = <DynArray<Item, 1> as AbsSquare>::Output;

    fn abs_square(&self) -> Self::Output {
        assert_eq!(
            self.index_layout.number_of_local_indices(),
            self.local.shape()[0]
        );

        let local_abs_square = self.local.abs_square();
        let comm = self.index_layout.comm();

        let mut global_result = Default::default();
        comm.all_reduce_into(
            &local_abs_square,
            &mut global_result,
            mpi::collective::SystemOperation::sum(),
        );
        global_result
    }
}

impl<'a, C, Item> NormSup for DistributedVector<'a, C, Item>
where
    C: Communicator,
    DynArray<Item, 1>: NormSup,
    <DynArray<Item, 1> as NormSup>::Output: Equivalence + Default,
{
    type Output = <DynArray<Item, 1> as NormSup>::Output;

    fn norm_sup(&self) -> Self::Output {
        assert_eq!(
            self.index_layout.number_of_local_indices(),
            self.local.shape()[0]
        );

        let local_norm_sup = self.local.norm_sup();
        let comm = self.index_layout.comm();

        let mut global_result = Default::default();
        comm.all_reduce_into(
            &local_norm_sup,
            &mut global_result,
            mpi::collective::SystemOperation::max(),
        );
        global_result
    }
}

impl<'a, C, Item> NormTwo for DistributedVector<'a, C, Item>
where
    C: Communicator,
    Self: AbsSquare,
    <Self as AbsSquare>::Output: Sqrt,
{
    type Output = <<Self as AbsSquare>::Output as Sqrt>::Output;

    fn norm_2(&self) -> Self::Output {
        Sqrt::sqrt(&self.abs_square())
    }
}
