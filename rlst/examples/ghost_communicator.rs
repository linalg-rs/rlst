//? mpirun -n 3

//! This examples demonstrates the use of the ghost communicator.
//!
//! We have three processes. The first process has indices 0, 1, 2, 3, 4.
//! The second process has indices 5, 6, 7, 8, 9. The third process has
//! indices 10, 11, 12, 13, 14.
//! The first process requires indices 5 and 6 from the second process as
//! ghost indices.
//! The second process requires index 4 from the third process as ghost index.
//! The third process requires indices 0, 1, 2 from the first process as ghost.

use mpi::traits::Communicator;
use rlst::distributed_tools::GhostCommunicator;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    // This example is designed for 3 MPI ranks.

    if world.size() != 3 {
        println!("Please run this example with 3 MPI ranks.");
        return;
    }

    // This example is designed for two processes.
    assert_eq!(
        world.size(),
        3,
        "This example is designed for three MPI ranks."
    );

    // We setup the ghost communicator.

    let ghost_comm = if rank == 0 {
        GhostCommunicator::new(&[5, 6], &[1, 1], &world)
    } else if rank == 1 {
        GhostCommunicator::new(&[10], &[2], &world)
    } else {
        GhostCommunicator::new(&[5, 0, 1, 2], &[1, 0, 0, 0], &world)
    };

    // We have now setup the ghost communicator.
    // Let us print the in-ranks and out-ranks for process 0,
    // and the receive_indices and send_indices.

    if rank == 2 {
        println!(
            "Process 1: In ranks: {:#?}, Out ranks: {:#?}, send_counts: {:#?}, receive_counts: {:#?}, receive_indices: {:#?}, send_indices: {:#?}",
            ghost_comm.in_ranks(),
            ghost_comm.out_ranks(),
            ghost_comm.send_counts(),
            ghost_comm.receive_counts(),
            ghost_comm.receive_indices(),
            ghost_comm.send_indices(),
        );
    }

    // Let us now send some data over the ghost communicator.

    let data = if rank == 0 {
        vec![10, 11, 12]
    } else if rank == 1 {
        vec![13, 14, 13]
    } else {
        vec![15]
    };

    let mut received_data = vec![0; ghost_comm.total_receive_count()];

    ghost_comm.forward_send_values(&data, &mut received_data);

    if rank == 0 {
        assert_eq!(received_data[0], 13);
        assert_eq!(received_data[1], 14);
    } else if rank == 1 {
        assert_eq!(received_data[0], 15);
    } else {
        assert_eq!(received_data[0], 10);
        assert_eq!(received_data[1], 11);
        assert_eq!(received_data[2], 12);
        assert_eq!(received_data[3], 13);
    }

    // We now want to send the received data back to the original owners.

    let mut send_data = vec![0; ghost_comm.total_send_count()];

    ghost_comm.backward_send_values(&received_data, &mut send_data);

    if rank == 0 {
        assert_eq!(send_data[0], 10);
        assert_eq!(send_data[1], 11);
        assert_eq!(send_data[2], 12);
    } else if rank == 1 {
        assert_eq!(send_data[0], 13);
        assert_eq!(send_data[1], 14);
        assert_eq!(send_data[2], 13);
    } else {
        assert_eq!(send_data[0], 15);
    }
}
