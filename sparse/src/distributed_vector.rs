//! An Indexable Vector is a container whose elements can be 1d indexed.
use crate::traits::index_layout::IndexLayout;

use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::*;
use rlst_common::types::{RlstResult, Scalar};
use rlst_dense::{rlst_col_vec, RawAccess, RawAccessMut};
use rlst_dense::{ColumnVectorD, RandomAccessByValue, Shape};

use crate::index_layout::DefaultMpiIndexLayout;

pub struct DistributedVector<'a, T: Scalar + Equivalence, C: Communicator> {
    index_layout: &'a DefaultMpiIndexLayout<'a, C>,
    local: ColumnVectorD<T>,
}

impl<'a, T: Scalar + Equivalence, C: Communicator> DistributedVector<'a, T, C> {
    pub fn new(index_layout: &'a DefaultMpiIndexLayout<'a, C>) -> Self {
        DistributedVector {
            index_layout,
            local: rlst_col_vec![T, index_layout.number_of_local_indices()],
        }
    }
    pub fn local(&self) -> &ColumnVectorD<T> {
        &self.local
    }

    pub fn local_mut(&mut self) -> &mut ColumnVectorD<T> {
        &mut self.local
    }

    pub fn to_root(&self) -> Option<ColumnVectorD<T>> {
        let comm = self.index_layout().comm();
        let root_process = comm.process_at_rank(0);

        if comm.rank() == 0 {
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
            let global_dim = self.index_layout().number_of_global_indices();

            let mut vec = rlst_col_vec![T, global_dim];
            //let mut vec = DefaultSerialVector::<T>::new(global_dim);

            let mut partition = PartitionMut::new(vec.data_mut(), counts, displacements);

            root_process.gather_varcount_into_root(self.local.data(), &mut partition);

            Some(vec)
        } else {
            root_process.gather_varcount_into(self.local.data());
            None
        }
    }

    pub fn from_root<LocalVec: RandomAccessByValue<Item = T> + Shape>(
        index_layout: &'a DefaultMpiIndexLayout<'a, C>,
        other: &Option<LocalVec>,
    ) -> RlstResult<Self> {
        let comm = index_layout.comm();
        let counts: Vec<i32> = (0..comm.size())
            .map(|index| {
                let index_range = index_layout.index_range(index as usize).unwrap();
                (index_range.1 - index_range.0) as i32
            })
            .collect();
        let displacements: Vec<i32> = (0..comm.size())
            .map(|index| {
                let index_range = index_layout.index_range(index as usize).unwrap();
                index_range.0 as i32
            })
            .collect();

        let mut distributed_vec = Self::new(index_layout);

        //let mut recvbuf = vec![T::zero(); index_layout.number_of_local_indices()];

        let root_process = comm.process_at_rank(0);
        if comm.rank() == 0 {
            assert!(other.is_some(), "`other` has a `none` value.");
            let other = other.as_ref().unwrap();
            assert!(
                other.shape().0 == 1 || other.shape().1 == 1,
                "`other` is not a vector"
            );
            let local_dim = other.shape().0 * other.shape().1;

            assert!(
                index_layout.number_of_global_indices() == local_dim,
                "Index Layout has {} dofs but `other` has {} elements",
                index_layout.number_of_global_indices(),
                local_dim
            );

            let mut data = vec![T::zero(); local_dim];
            for index in 0..local_dim {
                data[index] = other.get1d_value(index).unwrap();
            }
            let partition = Partition::new(data.as_slice(), counts, displacements);

            root_process.scatter_varcount_into_root(&partition, distributed_vec.local.data_mut());
        } else {
            assert!(other.is_none(), "`other` has a `Some` value.");
            root_process.scatter_varcount_into(distributed_vec.local_mut().data_mut());
        }

        Ok(distributed_vec)
    }

    fn index_layout(&self) -> &DefaultMpiIndexLayout<'a, C> {
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
