//! A distributed array is block-cyclic distributed collection of n-dimensional data across
//! multiple processes.

use mpi::traits::Communicator;

use crate::Array;

/// This struct describes the distribution of an n-dimensional array across processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockCyclicArrayDescriptor<const NDIM: usize> {
    /// The shape of the array.
    pub shape: [usize; NDIM],
    /// The block size in each dimension.
    pub block_size: [usize; NDIM],
    /// The process count in each dimension.
    pub nprocs: [usize; NDIM],
    /// The total number of processes.
    pub size: usize,
    /// The rank of the current process.
    pub rank: usize,
    /// The index of the current process.
    pub proc_index: [usize; NDIM],
    /// The local shape of the array on this process.
    pub local_shape: [usize; NDIM],
    /// The number of local blocks in each dimension.
    pub nlocal_blocks: [usize; NDIM],
}

impl<const NDIM: usize> BlockCyclicArrayDescriptor<NDIM> {
    /// Create a new distribution descriptor.
    pub fn new(
        shape: [usize; NDIM],
        block_size: [usize; NDIM],
        nprocs: [usize; NDIM],
        comm_size: usize,
        comm_rank: usize,
    ) -> Self {
        let proc_index = proc_index_from_rank(comm_rank, nprocs);
        let my_shape = local_shape(proc_index, nprocs, shape, block_size);

        let mut nlocal_blocks = [0; NDIM];

        for i in 0..NDIM {
            nlocal_blocks[i] = (my_shape[i] + block_size[i] - 1) / block_size[i];
        }

        Self {
            shape,
            block_size,
            nprocs,
            size: comm_size,
            rank: comm_rank,
            proc_index,
            local_shape: my_shape,
            nlocal_blocks,
        }
    }
}

/// Definition of a distributed array.
pub struct BlockCyclicArray<'a, C, ArrayImpl, const NDIM: usize> {
    comm: &'a C,
    local: Array<ArrayImpl, NDIM>,
    desc: BlockCyclicArrayDescriptor<NDIM>,
}

impl<'a, C, ArrayImpl, const NDIM: usize> BlockCyclicArray<'a, C, ArrayImpl, NDIM>
where
    C: Communicator,
{
    /// Create a new distributed array.
    pub(crate) fn new(
        comm: &'a C,
        local: Array<ArrayImpl, NDIM>,
        desc: BlockCyclicArrayDescriptor<NDIM>,
    ) -> Self {
        BlockCyclicArray { comm, local, desc }
    }
}

#[macro_export]
/// Create a new distributed array with given shape, block size, and process grid.
macro_rules! dist_array {
    ($ndim:literal, $dtype:ty, $shape:expr, $block_size:expr, $nprocs:expr, $comm:expr) => {{
        use mpi::traits::Communicator;
        let size = $comm.size() as usize;
        let rank = $comm.rank() as usize;
        assert_eq!(
            size,
            $nprocs.iter().product::<usize>(),
            "Number of processes does not match the product of `nprocs` dimensions."
        );
        let desc = $crate::dense::block_cyclic_array::BlockCyclicArrayDescriptor::new(
            $shape,
            $block_size,
            $nprocs,
            size,
            rank,
        );
        let local = $crate::dense::array::DynArray::<$dtype, $ndim>::from_shape(desc.local_shape);
        $crate::dense::block_cyclic_array::BlockCyclicArray::new($comm, local, desc)
    }};
}

/// This function is a port of the `numroc` function from ScaLAPACK.
/// It calculates the number of rows or columns that a process should own in a block-cyclic
/// distribution.
/// **Arguments:**
/// - `n`: The total number of rows or columns.
/// - `nb`: The block size.
/// - `iproc`: The rank of the process.
/// - `isrcproc`: The rank of the source process (the process that owns the first block).
/// - `nprocs`: The total number of processes.
/// **Returns:**
/// The number of rows or columns that the process `iproc` should own.
fn numroc(n: usize, nb: usize, iproc: usize, isrcproc: usize, nprocs: usize) -> usize {
    // The processes distance from the source process.
    let mydist = (nprocs + iproc - isrcproc) % nprocs;

    // The total number of whole blocks n is split into.
    let nblocks = n / nb;

    // The minimum number of entries a block can have.
    let mut numroc = (nblocks / nprocs) * nb;

    // Check if there are extra blocks
    let extrablks = nblocks % nprocs;

    if mydist < extrablks {
        // If this process is in the first `extrablks` processes, it gets an extra block.
        numroc += nb;
    } else if mydist == extrablks {
        // If this process is the last of the first `extrablks` processes, it gets the remainder.
        numroc += n % nb;
    }

    numroc
}

/// Calculate the nd index of the process from its rank.
fn proc_index_from_rank<const NDIM: usize>(rank: usize, nprocs: [usize; NDIM]) -> [usize; NDIM] {
    let proc_count = nprocs.iter().product::<usize>();
    assert!(rank < proc_count, "Rank out of bounds");
    let mut proc_index = [0; NDIM];
    let mut remaining_rank = rank;
    for i in 0..NDIM {
        proc_index[i] = remaining_rank % nprocs[i];
        remaining_rank /= nprocs[i];
    }
    proc_index
}

/// Calculate the number of local elements in each dimension.
fn local_shape<const NDIM: usize>(
    proc_index: [usize; NDIM],
    nprocs: [usize; NDIM],
    shape: [usize; NDIM],
    block_size: [usize; NDIM],
) -> [usize; NDIM] {
    let mut local_shape = [0; NDIM];
    for i in 0..NDIM {
        local_shape[i] = numroc(
            shape[i],
            block_size[i],
            proc_index[i],
            0, // Assuming the source process is always rank 0
            nprocs[i],
        );
    }
    local_shape
}

#[cfg(test)]
mod test {

    #[test]
    pub fn test() {
        use mpi;

        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let dist_array = dist_array!(2, f64, [100, 100], [10, 10], [1, 1], &world);
    }
}
