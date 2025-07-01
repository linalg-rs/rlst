//! Ghost Communicator
//!
//! This module provides a very simple ghost commucation framework on top of `rsmpi`.
//! One simply defines for each process the required ghost elements. The library then setups
//! a neighborhood communicator on each process that allows information to where it is needed
//! as ghost elements. A backward communicator is also implemented that allows a reeiver to
//! change a ghost and update the information back to the original process of the element.
//!
//! # Example
//!
//! A fully worked example is provided in the file `examples/ghost_communicator.rs`.

use std::os::raw::c_void;

use mpi_sys;

use mpi::topology::SimpleCommunicator;
use mpi::traits::{AsRaw, Communicator, CommunicatorCollectives, Equivalence, FromRaw};

/// Ghost communicator
pub struct GhostCommunicator<I: Default + Copy + Equivalence> {
    /// The `out` ranks that data is sent to from the current process.
    pub out_ranks: Vec<i32>,
    /// The `in` ranks that send data to the current process.
    pub in_ranks: Vec<i32>,
    /// Indices to send away
    pub send_indices: Vec<I>,
    /// Indices to be received
    pub receive_indices: Vec<I>,
    /// How many indices to send to the `out` vertices
    pub send_counts: Vec<i32>,
    /// How many indices to receive from the `in` vertices
    pub receive_counts: Vec<i32>,
    /// Neighbourhood send displacements
    pub send_displacements: Vec<i32>,
    /// Neighbourhood receive displacements
    pub receive_displacements: Vec<i32>,
    /// Total number of items to send
    pub total_send_count: usize,
    /// Total number of items to receive
    pub total_receive_count: usize,
    ///  The forward communicator
    pub forward_comm: SimpleCommunicator,
    /// The backward communicator that reverses the `in` and `out` vertices
    pub backward_comm: SimpleCommunicator,
}

impl<I: Default + Copy + Equivalence> GhostCommunicator<I> {
    /// Create new ghost communicator.
    ///
    /// # Arguments
    /// - `ghost_indices` - The ghost indices required on the current process.
    /// - `owning_ranks` - The ranks of the processes that own the ghost indices.
    /// - `comm` - The MPI communicator.
    pub fn new<C: Communicator>(
        ghost_indices: &[I],
        owning_ranks: &[usize],
        comm: &C,
    ) -> GhostCommunicator<I> {
        // Get the processes of global indices and create a map rank -> indices_on_rank

        let mut receive_counts = vec![0; comm.size() as usize];

        for &rank in owning_ranks {
            receive_counts[rank] += 1;
        }

        // Now sort the ghost indices by ranks
        // These are the receive indices, meaning the indices that we are receiving on the process.

        let receive_indices = {
            let mut sorted_ghost_index_args = (0..ghost_indices.len()).collect::<Vec<_>>();
            sorted_ghost_index_args.sort_by_key(|&i| owning_ranks[i]);

            let mut global_indices_t = Vec::<I>::with_capacity(ghost_indices.len());
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

        let mut out_ranks = Vec::<i32>::new();
        let mut in_ranks = Vec::<i32>::new();
        let mut receive_displacements = Vec::<i32>::new();
        let mut send_displacements = Vec::<i32>::new();

        // Get the in neighbours and the out-neighbours
        // for the neighborhood communicators. in-neighbors
        // are the ranks from which we receive data and out-neighbors
        // are the ranks to which we send data.

        let mut total_send_count = 0;
        let mut total_receive_count = 0;

        // We restrict the send and receive counts to the neighbors
        // and also compute the corresponding displacements.

        let (receive_counts, send_counts) = {
            let mut neighbor_receive_counts = Vec::<i32>::new();
            let mut neighbor_send_counts = Vec::<i32>::new();

            for index in 0..comm.size() as usize {
                if receive_counts[index] != 0 {
                    in_ranks.push(index as i32);
                    neighbor_receive_counts.push(receive_counts[index]);
                    receive_displacements.push(total_receive_count);
                    total_receive_count += receive_counts[index];
                }
                if send_counts[index] != 0 {
                    neighbor_send_counts.push(send_counts[index]);
                    out_ranks.push(index as i32);
                    send_displacements.push(total_send_count);
                    total_send_count += send_counts[index];
                }
            }
            (neighbor_receive_counts, neighbor_send_counts)
        };

        // To create the actual communicator need to call into mpi-sys as not yet wrapped into
        // higher level interface.

        let forward_comm = unsafe {
            let mut raw_comm = mpi_sys::RSMPI_COMM_NULL;
            mpi_sys::MPI_Dist_graph_create_adjacent(
                comm.as_raw(),
                in_ranks.len() as i32,
                in_ranks.as_ptr(),
                mpi_sys::RSMPI_UNWEIGHTED(),
                out_ranks.len() as i32,
                out_ranks.as_ptr(),
                mpi_sys::RSMPI_UNWEIGHTED(),
                mpi_sys::RSMPI_INFO_NULL,
                0,
                &mut raw_comm,
            );

            mpi::topology::SimpleCommunicator::from_raw(raw_comm)
        };

        // The backward communicator simply reverses the in and out neighbors.

        let backward_comm = unsafe {
            let mut raw_comm = mpi_sys::RSMPI_COMM_NULL;
            mpi_sys::MPI_Dist_graph_create_adjacent(
                comm.as_raw(),
                out_ranks.len() as i32,
                out_ranks.as_ptr(),
                mpi_sys::RSMPI_UNWEIGHTED(),
                in_ranks.len() as i32,
                in_ranks.as_ptr(),
                mpi_sys::RSMPI_UNWEIGHTED(),
                mpi_sys::RSMPI_INFO_NULL,
                0,
                &mut raw_comm,
            );

            mpi::topology::SimpleCommunicator::from_raw(raw_comm)
        };

        // We now communicate the global indices back from the receivers to the senders.

        let total_send_count = send_counts.iter().sum::<i32>() as usize;
        let total_receive_count = receive_counts.iter().sum::<i32>() as usize;

        let mut send_indices = vec![<I as Default>::default(); total_send_count];

        // The receivers know what indices they need from each process. But the
        // senders don't know yet what indices to send to each process. So we
        // send the receive indices back to the senders.

        unsafe {
            mpi_sys::MPI_Neighbor_alltoallv(
                receive_indices.as_ptr() as *const c_void,
                receive_counts.as_ptr(),
                receive_displacements.as_ptr(),
                <I as Equivalence>::equivalent_datatype().as_raw(),
                send_indices.as_mut_ptr() as *mut c_void,
                send_counts.as_ptr(),
                send_displacements.as_ptr(),
                <I as Equivalence>::equivalent_datatype().as_raw(),
                backward_comm.as_raw(),
            );
        }
        Self {
            out_ranks,
            in_ranks,
            send_indices,
            receive_indices,
            send_counts,
            receive_counts,
            send_displacements,
            receive_displacements,
            total_send_count,
            total_receive_count,
            forward_comm,
            backward_comm,
        }
    }

    /// Return the ranks to which the current process sends to.
    pub fn out_ranks(&self) -> &[i32] {
        &self.out_ranks
    }

    /// Return the ranks from which the current process receives from.
    pub fn in_ranks(&self) -> &[i32] {
        &self.in_ranks
    }

    /// Return the indices that are sent out from the current process.
    pub fn send_indices(&self) -> &[I] {
        &self.send_indices
    }

    /// Return the indices that are received on the current process.
    pub fn receive_indices(&self) -> &[I] {
        &self.receive_indices
    }

    /// Return the number of indices that are sent out to each process.
    pub fn send_counts(&self) -> &[i32] {
        &self.send_counts
    }

    /// Return the number of indices that are received from each process.
    pub fn receive_counts(&self) -> &[i32] {
        &self.receive_counts
    }

    /// Return the total send count.
    pub fn total_send_count(&self) -> usize {
        self.total_send_count
    }

    /// Return the total receive count.
    pub fn total_receive_count(&self) -> usize {
        self.total_receive_count
    }

    /// Return the forward communicator.
    ///
    /// This is a neighbourhood MPI communicator that sends values to the `out` processes
    /// and receives from the `in` processes.
    pub fn forward_comm(&self) -> &SimpleCommunicator {
        &self.forward_comm
    }

    /// Return the backward communicator.
    /// This is a neighbourhood MPI communicator that sends values to the `in` processes and
    /// receives values from the `out` processes.
    pub fn backward_comm(&self) -> &SimpleCommunicator {
        &self.backward_comm
    }

    /// Forward send values.
    ///
    /// This updates ghosts on the receiver process with the values of the ghosts from
    /// their owning process.
    pub fn forward_send_values<T: Equivalence>(&self, out_values: &[T], in_values: &mut [T]) {
        assert_eq!(in_values.len(), self.total_receive_count);
        assert_eq!(out_values.len(), self.total_send_count);

        unsafe {
            mpi_sys::MPI_Neighbor_alltoallv(
                out_values.as_ptr() as *const c_void,
                self.send_counts.as_ptr(),
                self.send_displacements.as_ptr(),
                <I as Equivalence>::equivalent_datatype().as_raw(),
                in_values.as_mut_ptr() as *mut c_void,
                self.receive_counts.as_ptr(),
                self.receive_displacements.as_ptr(),
                <I as Equivalence>::equivalent_datatype().as_raw(),
                self.forward_comm.as_raw(),
            );
        }
    }

    /// Forward send values with a given chunk size.
    ///
    /// This updates ghosts on the receiver process with the values of the ghosts from
    /// their owning process. In addition this method has a `chunk_size` parameter to communicate
    /// amounts of data in chunks larger than 1.
    pub fn forward_send_values_by_chunks<T: Equivalence>(
        &self,
        out_values: &[T],
        in_values: &mut [T],
        chunk_size: usize,
    ) {
        assert_eq!(in_values.len(), self.total_receive_count * chunk_size);
        assert_eq!(out_values.len(), self.total_send_count * chunk_size);

        let chunked_send_counts = self
            .send_counts
            .iter()
            .map(|&x| x * chunk_size as i32)
            .collect::<Vec<_>>();
        let chunked_receive_counts = self
            .receive_counts
            .iter()
            .map(|&x| x * chunk_size as i32)
            .collect::<Vec<_>>();
        let chunked_send_displacements = self
            .send_displacements
            .iter()
            .map(|&x| x * chunk_size as i32)
            .collect::<Vec<_>>();
        let chunked_receive_displacements = self
            .receive_displacements
            .iter()
            .map(|&x| x * chunk_size as i32)
            .collect::<Vec<_>>();

        unsafe {
            mpi_sys::MPI_Neighbor_alltoallv(
                out_values.as_ptr() as *const c_void,
                chunked_send_counts.as_ptr(),
                chunked_send_displacements.as_ptr(),
                <I as Equivalence>::equivalent_datatype().as_raw(),
                in_values.as_mut_ptr() as *mut c_void,
                chunked_receive_counts.as_ptr(),
                chunked_receive_displacements.as_ptr(),
                <I as Equivalence>::equivalent_datatype().as_raw(),
                self.forward_comm.as_raw(),
            );
        }
    }

    /// Backward send values.
    ///
    /// This back propagates updated ghost values from the receiver to the original owning process.
    pub fn backward_send_values<T: Equivalence>(&self, out_values: &[T], in_values: &mut [T]) {
        assert_eq!(out_values.len(), self.total_receive_count);
        assert_eq!(in_values.len(), self.total_send_count);

        unsafe {
            mpi_sys::MPI_Neighbor_alltoallv(
                out_values.as_ptr() as *const c_void,
                self.receive_counts.as_ptr(),
                self.receive_displacements.as_ptr(),
                <I as Equivalence>::equivalent_datatype().as_raw(),
                in_values.as_mut_ptr() as *mut c_void,
                self.send_counts.as_ptr(),
                self.send_displacements.as_ptr(),
                <I as Equivalence>::equivalent_datatype().as_raw(),
                self.backward_comm.as_raw(),
            );
        }
    }

    /// Backward send values.
    ///
    /// This back propagates updated ghost values from the receiver to the original owning process.
    /// In addition this method has a `chunk_size` parameter to communicate
    /// amounts of data in chunks larger than 1.
    pub fn backward_send_values_by_chunks<T: Equivalence>(
        &self,
        out_values: &[T],
        in_values: &mut [T],
        chunk_size: usize,
    ) {
        assert_eq!(out_values.len(), self.total_receive_count * chunk_size);
        assert_eq!(in_values.len(), self.total_send_count * chunk_size);

        let chunked_send_counts = self
            .send_counts
            .iter()
            .map(|&x| x * chunk_size as i32)
            .collect::<Vec<_>>();
        let chunked_receive_counts = self
            .receive_counts
            .iter()
            .map(|&x| x * chunk_size as i32)
            .collect::<Vec<_>>();
        let chunked_send_displacements = self
            .send_displacements
            .iter()
            .map(|&x| x * chunk_size as i32)
            .collect::<Vec<_>>();
        let chunked_receive_displacements = self
            .receive_displacements
            .iter()
            .map(|&x| x * chunk_size as i32)
            .collect::<Vec<_>>();

        unsafe {
            mpi_sys::MPI_Neighbor_alltoallv(
                out_values.as_ptr() as *const c_void,
                chunked_receive_counts.as_ptr(),
                chunked_receive_displacements.as_ptr(),
                <I as Equivalence>::equivalent_datatype().as_raw(),
                in_values.as_mut_ptr() as *mut c_void,
                chunked_send_counts.as_ptr(),
                chunked_send_displacements.as_ptr(),
                <I as Equivalence>::equivalent_datatype().as_raw(),
                self.backward_comm.as_raw(),
            );
        }
    }
}
