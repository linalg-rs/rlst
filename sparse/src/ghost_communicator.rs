//! Communication of ghost elements

use std::os::raw::c_void;

use mpi::topology::SimpleCommunicator;
use mpi::traits::{AsRaw, Communicator, CommunicatorCollectives, Equivalence, FromRaw};
use mpi_sys;
use rlst_dense::types::Scalar;

use crate::traits::index_layout::IndexLayout;

pub struct GhostCommunicator {
    pub global_receive_indices: Vec<usize>,
    pub local_send_indices: Vec<usize>,
    pub neighborhood_send_counts: Vec<i32>,
    pub neighborhood_receive_counts: Vec<i32>,
    pub neighborhood_send_displacements: Vec<i32>,
    pub neighborhood_receive_displacements: Vec<i32>,
    pub total_send_count: usize,
    pub total_receive_count: usize,
    neighbor_comm: SimpleCommunicator,
}

impl GhostCommunicator {
    pub fn new<C: Communicator, Layout: IndexLayout>(
        ghost_indices: &[usize],
        layout: &Layout,
        comm: &C,
    ) -> GhostCommunicator {
        // Get the processes of global indices and create a map rank -> indices_on_rank

        let mut ranks = Vec::<usize>::new();
        let mut receive_counts = vec![0; comm.size() as usize];
        let my_rank = comm.rank() as usize;

        for &ghost_index in ghost_indices {
            let rank = layout.rank_from_index(ghost_index).unwrap();
            assert_ne!(
                rank, my_rank,
                "Index {} already exists on rank {}.",
                ghost_index, my_rank
            );
            ranks.push(rank);
            receive_counts[rank] += 1;
        }

        // Now sort the global indices by ranks

        let global_receive_indices = {
            let mut sorted_ghost_index_args = (0..ghost_indices.len()).collect::<Vec<_>>();
            sorted_ghost_index_args.sort_by_key(|&i| ranks[i]);

            let mut global_indices_t = Vec::<usize>::with_capacity(ghost_indices.len());
            for arg in sorted_ghost_index_args {
                global_indices_t.push(ghost_indices[arg]);
            }
            global_indices_t
        };

        // We have now completed setting up the data on the ghost receivers. We now need
        // to tell the original owners of the ghosts who needs their data and setup the
        // communication structures. For this we first communicate the number indices via
        // an all_to_all.

        let mut send_counts = vec![0; comm.size() as usize];
        comm.all_to_all_into(&receive_counts, &mut send_counts);

        // Each process now has a list of ranks from which it receives and a list of indices
        // to which it sends. We now create a neighborhood communicator across all the ranks
        // from which a rank sends or receives

        // The following loop creates the neighbors, the receive displacements and the send
        // displacements.

        let mut neighbors = Vec::<i32>::new();
        let mut neighborhood_receive_counts: Vec<i32> = Vec::<i32>::new();
        let mut neighborhood_send_counts: Vec<i32> = Vec::<i32>::new();
        let mut neighborhood_receive_displacements: Vec<i32> = Vec::<i32>::new();
        let mut neighborhood_send_displacements: Vec<i32> = Vec::<i32>::new();

        let mut send_counter: i32 = 0;
        let mut receive_counter: i32 = 0;

        for index in 0..comm.size() as usize {
            if send_counts[index] != 0 || receive_counts[index] != 0 {
                neighbors.push(index as i32);
                neighborhood_send_counts.push(send_counts[index]);
                neighborhood_receive_counts.push(receive_counts[index]);
                neighborhood_send_displacements.push(send_counter);
                neighborhood_receive_displacements.push(receive_counter);
                send_counter += send_counts[index];
                receive_counter += receive_counts[index];
            }
        }

        let total_send_count = send_counter as usize;
        let total_receive_count = receive_counter as usize;

        // To create the actual communicator need to call into mpi-sys as not yet wrapped into
        // higher level interface.

        let neighbor_comm = unsafe {
            let mut raw_comm = mpi_sys::RSMPI_COMM_NULL;
            mpi_sys::MPI_Dist_graph_create_adjacent(
                comm.as_raw(),
                neighbors.len() as i32,
                neighbors.as_ptr(),
                mpi_sys::RSMPI_UNWEIGHTED(),
                neighbors.len() as i32,
                neighbors.as_ptr(),
                mpi_sys::RSMPI_UNWEIGHTED(),
                mpi_sys::RSMPI_INFO_NULL,
                0,
                &mut raw_comm,
            );

            mpi::topology::SimpleCommunicator::from_raw(raw_comm)
        };

        // We now communicate the global indices back from the receivers to the senders.

        let mut global_send_indices = vec![0; total_send_count];

        unsafe {
            mpi_sys::MPI_Neighbor_alltoallv(
                global_receive_indices.as_ptr() as *const c_void,
                neighborhood_receive_counts.as_ptr(),
                neighborhood_receive_displacements.as_ptr(),
                mpi_sys::RSMPI_UINT64_T,
                global_send_indices.as_mut_ptr() as *mut c_void,
                neighborhood_send_counts.as_ptr(),
                neighborhood_send_displacements.as_ptr(),
                mpi_sys::RSMPI_UINT64_T,
                neighbor_comm.as_raw(),
            );
        }

        Self {
            global_receive_indices,
            local_send_indices: global_send_indices
                .iter()
                .map(|&index| layout.global2local(my_rank, index).unwrap())
                .collect(),
            neighborhood_send_counts,
            neighborhood_receive_counts,
            neighborhood_send_displacements,
            neighborhood_receive_displacements,
            total_send_count,
            total_receive_count,
            neighbor_comm,
        }
    }

    pub fn forward_send_ghosts<T: Scalar + Equivalence>(
        &self,
        local_values: &[T],
        ghost_values: &mut [T],
    ) {
        assert_eq!(ghost_values.len(), self.total_receive_count);
        let mut send_values = Vec::<T>::with_capacity(self.local_send_indices.len());
        for &index in &self.local_send_indices {
            send_values.push(local_values[index]);
        }

        unsafe {
            mpi_sys::MPI_Neighbor_alltoallv(
                send_values.as_ptr() as *const c_void,
                self.neighborhood_send_counts.as_ptr(),
                self.neighborhood_send_displacements.as_ptr(),
                <T as Equivalence>::equivalent_datatype().as_raw(),
                ghost_values.as_mut_ptr() as *mut c_void,
                self.neighborhood_receive_counts.as_ptr(),
                self.neighborhood_receive_displacements.as_ptr(),
                <T as Equivalence>::equivalent_datatype().as_raw(),
                self.neighbor_comm.as_raw(),
            );
        }
    }
    pub fn backward_send_ghosts<Acc: Fn(&mut T, &T), T: Scalar + Equivalence>(
        &self,
        local_values: &mut [T],
        ghost_values: &[T],
        acc: Acc,
    ) {
        assert_eq!(ghost_values.len(), self.total_receive_count);

        let mut send_values = vec![T::zero(); self.total_send_count];

        unsafe {
            mpi_sys::MPI_Neighbor_alltoallv(
                ghost_values.as_ptr() as *const c_void,
                self.neighborhood_receive_counts.as_ptr(),
                self.neighborhood_receive_displacements.as_ptr(),
                <T as Equivalence>::equivalent_datatype().as_raw(),
                send_values.as_mut_ptr() as *mut c_void,
                self.neighborhood_send_counts.as_ptr(),
                self.neighborhood_send_displacements.as_ptr(),
                <T as Equivalence>::equivalent_datatype().as_raw(),
                self.neighbor_comm.as_raw(),
            );
        }

        for (&local_index, &value) in self.local_send_indices.iter().zip(send_values.iter()) {
            acc(&mut local_values[local_index], &value);
        }
    }
}
