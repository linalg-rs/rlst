//? mpirun -n {{NPROCESSES}} --features "mpi"

fn main() {
    pub use mpi::traits::*;
    pub use rlst_sparse::ghost_communicator::GhostCommunicator;
    pub use rlst_sparse::index_layout::DefaultMpiIndexLayout;
    pub use rlst_sparse::traits::index_layout;

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let n = 5;

    let index_layout = DefaultMpiIndexLayout::new(n, &world);

    let (global_indices, my_values) = {
        if rank == 0 {
            (vec![1], vec![1.0])
        } else {
            (Vec::<usize>::new(), vec![(1 + rank) as f64])
        }
    };

    let gc = GhostCommunicator::new(&global_indices, &index_layout, &world);

    let mut ghosts = vec![0.0; gc.total_receive_count];

    gc.forward_send_ghosts(&my_values, &mut ghosts);

    if rank == 0 {
        println!("{:#?}", ghosts);
    }
}
