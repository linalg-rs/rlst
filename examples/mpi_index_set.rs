//! Setup an index range.

use rlst::prelude::*;

use mpi::traits::Communicator;

const NCHUNKS: usize = 2;
const CHUNK_SIZE: usize = 3;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let index_layout = DefaultDistributedIndexLayout::new(NCHUNKS, CHUNK_SIZE, &world);

    if world.rank() == 0 {
        println!("Local index range: {:#?}", index_layout.local_range());
    }
}
