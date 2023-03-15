//! Various tools

use mpi::traits::*;
use rlst_traits::types::IndexType;

/// Check if an Option has a ```Some``` value on exactly one process (typically root).
///
/// If true return Some(index), where index is the rank of the ```Some```. Otherwise,
/// return ```None```.
pub fn has_unique_some<T, C: Communicator>(param: &Option<T>, comm: &C) -> Option<i32> {
    let flag: i32 = match param {
        Some(_) => 1,
        None => 0,
    };

    let mut global_result: i32 = 0;

    // Send around the flag. At the end the sum of
    // all flags should be 1.
    comm.all_reduce_into(
        &flag,
        &mut global_result,
        mpi::collective::SystemOperation::sum(),
    );

    if global_result != 1 {
        return None;
    }

    // We now know that the param has a value
    // only on one process. Now let's send around
    // the rank index to everyone.

    let flag: IndexType = match param {
        Some(_) => comm.rank() as IndexType,
        None => 0,
    };

    global_result = 0;

    comm.all_reduce_into(
        &flag,
        &mut global_result,
        mpi::collective::SystemOperation::sum(),
    );

    Some(global_result)
}
