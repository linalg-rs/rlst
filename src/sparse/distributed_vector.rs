//! An Indexable Vector is a container whose elements can be 1d indexed.
use std::cell::{Ref, RefCell, RefMut};

use crate::sparse::traits::index_layout::IndexLayout;

use crate::dense::array::DynamicArray;
use crate::dense::traits::{RawAccess, RawAccessMut, Shape};
use crate::dense::types::RlstScalar;
use crate::{rlst_dynamic_array1, Array, UnsafeRandomAccessByValue, UnsafeRandomAccessMut};
use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence, Root};
use mpi::Rank;

use crate::sparse::index_layout::DefaultMpiIndexLayout;

/// Distributed vector
pub struct DistributedVector<'a, Item: RlstScalar + Equivalence, C: Communicator> {
    index_layout: &'a DefaultMpiIndexLayout<'a, C>,
    local: RefCell<DynamicArray<Item, 1>>, // A RefCell is necessary as we often need a reference to the communicator and mutable ref to local at the same time.
                                           // But this would be disallowed by Rust's static borrow checker.
}

impl<'a, Item: RlstScalar + Equivalence, C: Communicator> DistributedVector<'a, Item, C> {
    /// Crate new
    pub fn new(index_layout: &'a DefaultMpiIndexLayout<'a, C>) -> Self {
        DistributedVector {
            index_layout,
            local: RefCell::new(rlst_dynamic_array1!(
                Item,
                [index_layout.number_of_local_indices()]
            )),
        }
    }
    /// Local part
    pub fn local(&self) -> Ref<'_, DynamicArray<Item, 1>> {
        //Array<Item, ArrayView<'_, Item, BaseArray<Item, VectorContainer<Item>, 1>, 1>, 1> {
        self.local.borrow()
    }

    /// Mutable local part
    pub fn local_mut(&self) -> RefMut<DynamicArray<Item, 1>> {
        // The data is behind a RefCell, so do not require mutable reference to self.
        self.local.borrow_mut()
    }

    /// Gather `Self` to all processes and store in `arr`.
    pub fn gather_to_all<
        ArrayImpl: UnsafeRandomAccessByValue<1, Item = Item>
            + Shape<1>
            + UnsafeRandomAccessMut<1>
            + RawAccessMut<Item = Item>,
    >(
        &self,
        mut arr: Array<Item, ArrayImpl, 1>,
    ) {
        let comm = self.index_layout.comm();
        assert_eq!(self.index_layout.number_of_global_indices(), arr.shape()[0]);

        let this_process = comm.this_process();

        let counts: Vec<i32> = (0..comm.size())
            .map(|index| {
                let index_range = self.index_layout.index_range(index as usize).unwrap();
                (index_range.1 - index_range.0) as i32
            })
            .collect();
        let displacements: Vec<i32> = (0..comm.size())
            .map(|index| {
                let index_range = self.index_layout.index_range(index as usize).unwrap();
                index_range.0 as i32
            })
            .collect();

        let mut partition = PartitionMut::new(arr.data_mut(), counts, displacements);

        this_process.all_gather_varcount_into(self.local().data(), &mut partition);
    }

    /// Send vector to a given rank (call this from the root)
    pub fn gather_to_rank_root<
        ArrayImpl: UnsafeRandomAccessByValue<1, Item = Item>
            + Shape<1>
            + UnsafeRandomAccessMut<1>
            + RawAccessMut<Item = Item>,
    >(
        &self,
        mut arr: Array<Item, ArrayImpl, 1>,
    ) {
        let comm = self.index_layout().comm();
        let my_process = comm.this_process();

        let global_dim = self.index_layout().number_of_global_indices();

        assert_eq!(arr.shape()[0], global_dim);

        let counts: Vec<i32> = (0..comm.size())
            .map(|index| {
                let index_range = self.index_layout.index_range(index as usize).unwrap();
                (index_range.1 - index_range.0) as i32
            })
            .collect();

        let displacements: Vec<i32> = (0..comm.size())
            .map(|index| {
                let index_range = self.index_layout.index_range(index as usize).unwrap();
                index_range.0 as i32
            })
            .collect();

        let mut partition = PartitionMut::new(arr.data_mut(), counts, displacements);

        my_process.gather_varcount_into_root(self.local().data(), &mut partition);
    }

    /// Send vector to a given rank (call this from the root)
    pub fn gather_to_rank(&self, target_rank: usize) {
        let target_process = self
            .index_layout()
            .comm()
            .process_at_rank(target_rank as Rank);

        target_process.gather_varcount_into(self.local().data());
    }

    /// Create from root
    pub fn scatter_from_root<
        ArrayImpl: UnsafeRandomAccessByValue<1, Item = Item>
            + Shape<1>
            + UnsafeRandomAccessMut<1>
            + RawAccessMut<Item = Item>,
    >(
        &mut self,
        arr: Array<Item, ArrayImpl, 1>,
    ) {
        let comm = self.index_layout.comm();

        let root_process = comm.this_process();

        assert_eq!(arr.shape()[0], self.index_layout.number_of_global_indices());
        let counts: Vec<i32> = (0..comm.size())
            .map(|index| {
                let index_range = self.index_layout.index_range(index as usize).unwrap();
                (index_range.1 - index_range.0) as i32
            })
            .collect();
        let displacements: Vec<i32> = (0..comm.size())
            .map(|index| {
                let index_range = self.index_layout.index_range(index as usize).unwrap();
                index_range.0 as i32
            })
            .collect();

        let partition = Partition::new(arr.data(), counts, displacements);

        root_process.scatter_varcount_into_root(&partition, self.local_mut().data_mut());
    }

    /// Create from root
    pub fn scatter_from(&mut self, source_rank: usize) {
        let source_process = self
            .index_layout()
            .comm()
            .process_at_rank(source_rank as Rank);

        source_process.scatter_varcount_into(self.local_mut().data_mut());
    }

    /// Return the index layout.
    pub fn index_layout(&self) -> &DefaultMpiIndexLayout<'a, C> {
        self.index_layout
    }
}

// impl<'a, T: Scalar + Equivalence, C: Communicator> IndexableVector for DefaultMpiVector<'a, T, C> {
//     type Item = T;
//     type View<'b> = LocalIndexableVectorView<'b, T> where Self: 'b;
//     type ViewMut<'b> = LocalIndexableVectorViewMut<'b, T> where Self: 'b;
//     type Ind = DefaultMpiIndexLayout<'a, C>;

//     fn index_layout(&self) -> &Self::Ind {
//         &self.index_layout
//     }

//     fn view<'b>(&'b self) -> Option<Self::View<'b>> {
//         Some(rlst_dense::RawAccess::data(&self.local))
//     }

//     fn view_mut<'b>(&'b mut self) -> Option<Self::ViewMut<'b>> {
//         Some(self.local.view_mut().unwrap())
//     }
// }

// impl<T: Scalar + Equivalence, C: Communicator> Inner for DefaultMpiVector<'_, T, C> {
//     fn inner(&self, other: &Self) -> RlstResult<Self::Item> {
//         let result;

//         if let Ok(local_result) = self.local.inner(&other.local()) {
//             result = local_result;
//         } else {
//             panic!(
//                 "Could not perform local inner product on process {}",
//                 self.index_layout().comm().rank()
//             );
//         }

//         let comm = self.index_layout.comm();

//         let mut global_result = T::zero();
//         comm.all_reduce_into(
//             &result,
//             &mut global_result,
//             mpi::collective::SystemOperation::sum(),
//         );
//         Ok(global_result)
//     }
// }

// impl<T: Scalar + Equivalence, C: Communicator> AbsSquareSum for DefaultMpiVector<'_, T, C>
// where
//     T::Real: Equivalence,
// {
//     fn abs_square_sum(&self) -> <Self::Item as Scalar>::Real {
//         let comm = self.index_layout.comm();

//         let local_result = self.local.abs_square_sum();

//         let mut global_result = <<Self::Item as Scalar>::Real>::zero();
//         comm.all_reduce_into(
//             &local_result,
//             &mut global_result,
//             mpi::collective::SystemOperation::sum(),
//         );
//         global_result
//     }
// }

// impl<T: Scalar + Equivalence, C: Communicator> Norm1 for DefaultMpiVector<'_, T, C>
// where
//     T::Real: Equivalence,
// {
//     fn norm_1(&self) -> <Self::Item as Scalar>::Real {
//         let comm = self.index_layout.comm();

//         let local_result = self.local.norm_1();

//         let mut global_result = <<Self::Item as Scalar>::Real as Zero>::zero();
//         comm.all_reduce_into(
//             &local_result,
//             &mut global_result,
//             mpi::collective::SystemOperation::sum(),
//         );
//         global_result
//     }
// }

// impl<T: Scalar + Equivalence, C: Communicator> Norm2 for DefaultMpiVector<'_, T, C>
// where
//     T::Real: Equivalence,
// {
//     fn norm_2(&self) -> <Self::Item as Scalar>::Real {
//         Float::sqrt(self.abs_square_sum())
//     }
// }

// impl<T: Scalar + Equivalence, C: Communicator> NormInfty for DefaultMpiVector<'_, T, C>
// where
//     T::Real: Equivalence,
// {
//     fn norm_infty(&self) -> <Self::Item as Scalar>::Real {
//         let comm = self.index_layout.comm();

//         let local_result = self.local.norm_infty();

//         let mut global_result = <<Self::Item as Scalar>::Real as Zero>::zero();
//         comm.all_reduce_into(
//             &local_result,
//             &mut global_result,
//             mpi::collective::SystemOperation::max(),
//         );
//         global_result
//     }
// }
