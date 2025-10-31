//! Maps global data to local data required by processes.
//!
//! Consider ten global dofs with indices 0 to 9 and two processes. The first process
//! may need the dofs 0, 1, 2, 3, 4, 5, 6 and the second process the dofs 3, 4, 5, 6, 7, 8, 9.
//! Hence, some dofs are needed on both processes. The `Global2LocalDataMapper` establishes the corresponding
//! communication and maps distributed vectors of global dofs to the required dofs on each process.

use std::{collections::HashMap, rc::Rc};

use itertools::{Itertools, izip};
use mpi::traits::{Communicator, Equivalence};

use super::IndexLayout;

/// Maps global data to local data.
pub struct Global2LocalDataMapper<'a, C: Communicator> {
    index_layout: Rc<IndexLayout<'a, C>>,
    ghost_communicator: super::GhostCommunicator<usize>,
    ghost_to_position: HashMap<usize, usize>,
    required_dofs: Vec<usize>,
}

impl<'a, C: Communicator> Global2LocalDataMapper<'a, C> {
    /// Create a new data mapper.
    ///
    /// The `required_dofs` are the dofs that are required on the local process.
    pub fn new(index_layout: Rc<IndexLayout<'a, C>>, required_dofs: &[usize]) -> Self {
        let comm = index_layout.comm();
        let rank = comm.rank() as usize;

        // First we go through the required dofs and get the ghosts

        let mut ghost_dofs = Vec::<usize>::new();

        for &dof in required_dofs.iter() {
            if index_layout.rank_from_index(dof).unwrap() != rank {
                ghost_dofs.push(dof);
            }
        }

        // Now make sure that the ghosts are unique

        let ghost_dofs = ghost_dofs.iter().unique().copied().collect_vec();

        // Get the ranks of the ghost

        let ghost_owners = ghost_dofs
            .iter()
            .map(|&dof| index_layout.rank_from_index(dof).unwrap())
            .collect_vec();

        // We now setup the ghost communicator

        let ghost_communicator = super::GhostCommunicator::new(&ghost_dofs, &ghost_owners, comm);

        // We now create a dof to position array for the ghosts

        let ghost_to_position = HashMap::<usize, usize>::from_iter(
            ghost_communicator
                .receive_indices
                .iter()
                .enumerate()
                .map(|(i, &d)| (d, i)),
        );

        Self {
            index_layout,
            ghost_communicator,
            ghost_to_position,
            required_dofs: required_dofs.to_vec(),
        }
    }

    /// Map global data to the local required data
    ///
    /// The input data is a vector of global data. A chunk size can be given in case multiple elements
    /// are associated with each dof.
    pub fn map_data<T: Equivalence + Copy + std::fmt::Debug>(
        &self,
        data: &[T],
        chunk_size: usize,
    ) -> Vec<T> {
        // First we need to go through the send dofs and set up the data that needs to be sent.

        let rank = self.index_layout.comm().rank() as usize;

        // Prepare the send data

        let send_data = {
            let mut tmp =
                Vec::<T>::with_capacity(self.ghost_communicator.total_send_count() * chunk_size);
            let send_buffer: &mut [T] = unsafe { std::mem::transmute(tmp.spare_capacity_mut()) };
            for (global_send_index, send_buffer_chunk) in izip!(
                self.ghost_communicator.send_indices().iter(),
                send_buffer.chunks_mut(chunk_size)
            ) {
                let local_start_index = self
                    .index_layout
                    .global2local(rank, *global_send_index)
                    .unwrap()
                    * chunk_size;
                let local_end_index = local_start_index + chunk_size;
                send_buffer_chunk.copy_from_slice(&data[local_start_index..local_end_index]);
            }
            unsafe { tmp.set_len(self.ghost_communicator.total_send_count() * chunk_size) };
            tmp
        };

        // Now get the receive data

        let receive_data = {
            let mut tmp =
                Vec::<T>::with_capacity(self.ghost_communicator.total_receive_count() * chunk_size);
            let receive_buffer: &mut [T] = unsafe { std::mem::transmute(tmp.spare_capacity_mut()) };
            self.ghost_communicator.forward_send_values_by_chunks(
                &send_data,
                receive_buffer,
                chunk_size,
            );
            unsafe { tmp.set_len(self.ghost_communicator.total_receive_count() * chunk_size) };
            tmp
        };

        // We have the receive data from the other processes. We now need to setup the output vector
        // and collect the data from what is already on the process and from the ghosts.

        {
            let total_number_of_dofs = self.required_dofs.len();
            let mut output_data = Vec::<T>::with_capacity(total_number_of_dofs * chunk_size);

            let output_buffer: &mut [T] =
                unsafe { std::mem::transmute(output_data.spare_capacity_mut()) };

            // We go through the dofs one by one, check whether it is owned or a ghost
            // and copy over the corresponding data

            for (&dof, output_chunk) in izip!(
                self.required_dofs.iter(),
                output_buffer.chunks_mut(chunk_size)
            ) {
                if self.index_layout.rank_from_index(dof).unwrap() == rank {
                    // Dof is owned. Need to copy the corresponding data
                    let local_dof = self.index_layout.global2local(rank, dof).unwrap();
                    let local_data_start = local_dof * chunk_size;
                    let local_data_end = local_data_start + chunk_size;

                    output_chunk.copy_from_slice(&data[local_data_start..local_data_end]);
                } else {
                    // Dof is a ghost Need to copy from ghosts
                    let receive_start = self.ghost_to_position[&dof] * chunk_size;
                    let receive_end = receive_start + chunk_size;

                    output_chunk.copy_from_slice(&receive_data[receive_start..receive_end]);
                }
            }

            unsafe { output_data.set_len(total_number_of_dofs * chunk_size) };
            output_data
        }
    }

    /// Return the index layout
    pub fn index_layout(&self) -> Rc<IndexLayout<'a, C>> {
        self.index_layout.clone()
    }

    /// Return the ghost communicator
    pub fn ghost_communicator(&self) -> &super::GhostCommunicator<usize> {
        &self.ghost_communicator
    }
}
