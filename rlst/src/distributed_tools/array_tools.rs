//! Array tools
//!
//! This module contains tools for working with distributed arrays.

use itertools::{Itertools, izip};
use mpi::{
    collective::{SystemOperation, UserOperation},
    datatype::PartitionMut,
    point_to_point::send_receive,
    traits::{Communicator, CommunicatorCollectives, Destination, Equivalence, Root, Source},
};

use crate::{Max, Min, TotalCmp};

///
/// Distribute a sorted sequence into bins.
///
/// For an array with n elements to be distributed into p bins,
/// the array `bins` has p elements. The bins are defined by half-open intervals
/// of the form [b_j, b_{j+1})). The final bin is the half-open interval [b_{p-1}, \infty).
/// It is assumed that the bins and the elements are both sorted sequences and that
/// every element has an associated bin.
/// The function returns a p element array with the counts of how many elements go to each bin.
/// Since the sequence is sorted this fully defines what element goes into which bin.
pub fn sort_to_bins<T: TotalCmp>(sorted_keys: &[T], bins: &[T]) -> Vec<usize> {
    let nbins = bins.len();

    // Deal with the special case that there is only one bin.
    // This means that all elements are in the one bin.
    if nbins == 1 {
        return vec![sorted_keys.len(); 1];
    }

    let mut bin_counts = vec![0; nbins];

    // This iterates over each possible bin and returns also the associated rank.
    // The last bin position is not iterated over since for an array with p elements
    // there are p-1 tuple windows.
    let mut bin_iter = izip!(
        bin_counts.iter_mut(),
        bins.iter().tuple_windows::<(&T, &T)>(),
    );

    // We take the first element of the bin iterator. There will always be at least one since
    // there are at least two bins (an actual one, and the last half infinite one)
    let mut r: &mut usize;
    let mut bin_start: &T;
    let mut bin_end: &T;
    (r, (bin_start, bin_end)) = bin_iter.next().unwrap();

    let mut count = 0;
    'outer: for key in sorted_keys.iter() {
        if bin_start.le(*key) && key.lt(*bin_end) {
            *r += 1;
            count += 1;
        } else {
            // Move the bin forward until it fits. There will always be a fitting bin.
            loop {
                if let Some((rn, (bsn, ben))) = bin_iter.next() {
                    if bsn.le(*key) && key.lt(*ben) {
                        // We have found the next fitting bin for our current element.
                        // Can register it and go back to the outer for loop.
                        *rn += 1;
                        r = rn;
                        bin_start = bsn;
                        bin_end = ben;
                        count += 1;
                        break;
                    }
                } else {
                    // We have no more fitting bin. So break the outer loop.
                    break 'outer;
                }
            }
        }
    }

    // We now have everything but the last bin. Just bunch the remaining elements to
    // the last count.
    *bin_counts.last_mut().unwrap() = sorted_keys.len() - count;

    bin_counts
}

/// Compute displacements on n processes from a vector of n counts of local indices.
///
/// This is useful for global MPI varcount operations. Let
/// count be `[ 3, 4, 5]`. Then the corresponding displacements are
// [0, 3, 7]. Note that the last element `5` is ignored for the displacements.
pub fn displacements(counts: &[i32]) -> Vec<i32> {
    counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect()
}

/// Performs an all-to-all communication.
///
/// # Input arguments
/// - `comm` - The communicator
/// - `counts` - A slice with `comm.size()` elements specifying how many elements to send to each process from the current process.
/// - `data` - The buffer of data to be sent out ordered with respect to `counts`.
///
/// The returned data is a tuple `(in_counts, in_data)` with `in_counts` an array with `comm.size()` elements specifying how
/// many elements have been received into the current process from each other process. `in_data` contains the actual received
/// data sorted according to `in_counts`.
pub fn all_to_allv<T: Equivalence>(
    comm: &impl Communicator,
    counts: &[usize],
    data: &[T],
) -> (Vec<usize>, Vec<T>) {
    // We need the counts as i32 types.
    assert_eq!(counts.len(), comm.size() as usize);

    let counts = counts.iter().map(|&x| x as i32).collect_vec();

    // First send around the counts via an all-to-all
    let mut recv_counts = vec![0; comm.size() as usize];
    comm.all_to_all_into(&counts, &mut recv_counts);

    let n_recv_counts = recv_counts.iter().sum::<i32>() as usize;

    // Now we can prepare the actual data. We have to allocate the data and compute the send partition and the receive partition.

    let mut receive_data = Vec::<T>::with_capacity(n_recv_counts);
    let receive_buf: &mut [T] = unsafe { std::mem::transmute(receive_data.spare_capacity_mut()) };

    let send_displacements = displacements(&counts);

    let receive_displacements = displacements(&recv_counts);

    let send_partition = mpi::datatype::Partition::new(data, counts, send_displacements);
    let mut receive_partition =
        mpi::datatype::PartitionMut::new(receive_buf, &recv_counts[..], receive_displacements);

    comm.all_to_all_varcount_into(&send_partition, &mut receive_partition);

    unsafe { receive_data.set_len(n_recv_counts) };

    (
        recv_counts.iter().map(|i| *i as usize).collect_vec(),
        receive_data,
    )
}

/// Scatter data across processes.
///
/// This function needs to be called at the root for the scatter operation.
/// # Input arguments
/// - `comm` - The communicator
/// - `counts` - A slice of length `comm.size()` containing the number of elements to be sent to each process.
/// - `out_data` - The data to be sent out sorted by `counts`.
///
/// The function returns the vector of elements that is sent to the root in the scatter operation.
pub fn scatterv_root<T: Equivalence>(
    comm: &impl Communicator,
    counts: &[usize],
    out_data: &[T],
) -> Vec<T> {
    assert_eq!(counts.len(), comm.size() as usize);
    let rank = comm.rank() as usize;

    let send_counts = counts.iter().map(|&x| x as i32).collect_vec();

    let mut recv_count: i32 = 0;
    let mut recvbuf: Vec<T> = Vec::<T>::with_capacity(send_counts[rank] as usize);
    // This avoids having the pre-initialise the array. We simply transmute the spare capacity
    // into a valid reference and later manually set the length of the array to the full capacity.
    let recvbuf_ref: &mut [T] = unsafe { std::mem::transmute(recvbuf.spare_capacity_mut()) };

    let displacements = displacements(&send_counts);

    // Now scatter the counts to each process.
    comm.this_process()
        .scatter_into_root(&send_counts, &mut recv_count);

    // We now prepare the send partition of the variable length data.
    let send_partition =
        mpi::datatype::Partition::new(out_data, &send_counts[..], &displacements[..]);

    // And now we send the partition.
    comm.this_process()
        .scatter_varcount_into_root(&send_partition, recvbuf_ref);

    unsafe { recvbuf.set_len(send_counts[rank] as usize) };

    recvbuf
}

/// Receive the scattered data from `root`.
pub fn scatterv<T: Equivalence + Copy>(comm: &impl Communicator, root: usize) -> Vec<T> {
    let mut recv_count: i32 = 0;

    // First we need to receive the number of elements that we are about to get.
    comm.process_at_rank(root as i32)
        .scatter_into(&mut recv_count);

    // We prepare an unitialized buffer to receive the data.
    let mut recvbuf: Vec<T> = Vec::<T>::with_capacity(recv_count as usize);
    // This avoids having the pre-initialise the array. We simply transmute the spare capacity
    // into a valid reference and later manually set the length of the array to the full capacity.
    let recvbuf_ref: &mut [T] = unsafe { std::mem::transmute(recvbuf.spare_capacity_mut()) };

    // And finally we receive the data.
    comm.process_at_rank(root as i32)
        .scatter_varcount_into(recvbuf_ref);

    // Don't forget to manually set the length of the vector to the correct value.
    unsafe { recvbuf.set_len(recv_count as usize) };
    recvbuf
}

/// Get the minimum value across all ranks
pub fn global_min<T: Equivalence + Copy + Min<Output = T>, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> T {
    let local_min = arr
        .iter()
        .copied()
        .reduce(|x, y| x.min(y))
        .unwrap_or_else(|| {
            panic!(
                "global_min: Local array on process {} is empty.",
                comm.rank()
            )
        });

    // Just need to initialize global_min with something.
    let mut global_min = local_min;

    comm.all_reduce_into(
        &local_min,
        &mut global_min,
        &UserOperation::commutative(|x, y| {
            let x: &[T] = x.downcast().unwrap();
            let y: &mut [T] = y.downcast().unwrap();
            for (&x_i, y_i) in x.iter().zip(y) {
                *y_i = x_i.min(*y_i);
            }
        }),
    );

    global_min
}

/// Get the maximum value across all ranks
pub fn global_max<T: Equivalence + Copy + Max<Output = T>, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> T {
    let local_max = arr
        .iter()
        .copied()
        .reduce(|x, y| x.max(y))
        .unwrap_or_else(|| {
            panic!(
                "global_max: Local array on process {} is empty.",
                comm.rank()
            )
        });

    // Just need to initialize global_max with something.
    let mut global_max = local_max;

    comm.all_reduce_into(
        &local_max,
        &mut global_max,
        &UserOperation::commutative(|x, y| {
            let x: &[T] = x.downcast().unwrap();
            let y: &mut [T] = y.downcast().unwrap();
            for (&x_i, y_i) in x.iter().zip(y) {
                *y_i = x_i.max(*y_i);
            }
        }),
    );

    global_max
}

/// Gather array to all processes
pub fn gather_to_all<T: Equivalence, C: CommunicatorCollectives>(arr: &[T], comm: &C) -> Vec<T> {
    // First we need to broadcast the individual sizes on each process.

    let size = comm.size();

    let local_len = arr.len() as i32;

    let mut sizes = vec![0; size as usize];

    comm.all_gather_into(&local_len, &mut sizes);

    let recv_len = sizes.iter().sum::<i32>() as usize;

    let mut recvbuffer = Vec::<T>::with_capacity(recv_len);
    let buf: &mut [T] = unsafe { std::mem::transmute(recvbuffer.spare_capacity_mut()) };

    let recv_displs: Vec<i32> = displacements(&sizes);

    let mut receiv_partition = PartitionMut::new(buf, sizes, &recv_displs[..]);

    comm.all_gather_varcount_into(arr, &mut receiv_partition);

    unsafe { recvbuffer.set_len(recv_len) };

    recvbuffer
}

/// Communicate the first element of each local array back to the previous rank and
/// return this result on each rank.
///
/// The last rank returns `None`. The other ranks return the first value in the array
/// of the next process.
pub fn communicate_back<T: Equivalence, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> Option<T> {
    let rank = comm.rank();
    let size = comm.size();

    if size == 1 {
        return None;
    }

    if rank == size - 1 {
        comm.process_at_rank(rank - 1).send(arr.first().unwrap());
        None
    } else {
        let (new_last, _status) = if rank > 0 {
            send_receive(
                arr.first().unwrap_or_else(|| {
                    panic!("communicate_back: Array on process {} is empty", rank)
                }),
                &comm.process_at_rank(rank - 1),
                &comm.process_at_rank(rank + 1),
            )
        } else {
            comm.process_at_rank(1).receive::<T>()
        };
        Some(new_last)
    }
}

/// Check if a distributed array is sorted.
pub fn is_sorted_array<T: Equivalence + TotalCmp, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> bool {
    let mut sorted = true;
    for (elem1, elem2) in arr.iter().tuple_windows() {
        if elem1.gt(*elem2) {
            sorted = false;
        }
    }

    if comm.size() == 1 {
        return sorted;
    }

    if let Some(next_first) = communicate_back(arr, comm) {
        sorted = sorted
            && arr
                .last()
                .unwrap_or_else(|| {
                    panic!(
                        "is_sorted_array: Array on process {} is empty.",
                        comm.rank()
                    )
                })
                .le(next_first);
    }

    let mut global_sorted: bool = false;
    comm.all_reduce_into(&sorted, &mut global_sorted, SystemOperation::logical_and());

    global_sorted
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::distributed_tools::sort_to_bins;

    #[test]
    fn test_sort_to_bins() {
        let arr = (1..98).collect_vec();
        let bins = vec![1, 10, 20, 30, 40, 50, 60, 70, 80, 90];

        let counts = sort_to_bins(&arr, &bins);

        assert_eq!(counts[0], 9);
        assert_eq!(counts[1], 10);
        assert_eq!(counts[2], 10);
        assert_eq!(counts[3], 10);
        assert_eq!(counts[4], 10);
        assert_eq!(counts[5], 10);
        assert_eq!(counts[6], 10);
        assert_eq!(counts[7], 10);
        assert_eq!(counts[8], 10);
        assert_eq!(counts[9], 8);

        assert_eq!(counts.iter().sum::<usize>(), arr.len());

        let arr = vec![15];

        let counts = sort_to_bins(&arr, &bins);

        assert_eq!(counts.iter().sum::<usize>(), arr.len());

        assert_eq!(counts[1], 1);

        let arr = vec![99];

        let counts = sort_to_bins(&arr, &bins);
        assert_eq!(counts.iter().sum::<usize>(), arr.len());

        assert_eq!(counts[9], 1);
    }
}
