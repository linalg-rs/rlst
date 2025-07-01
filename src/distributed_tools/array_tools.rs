//! Array tools
//!
//! This module contains tools for working with distributed arrays.

use itertools::{izip, Itertools};
use mpi::{
    datatype::{Partition, PartitionMut},
    traits::{Communicator, CommunicatorCollectives, Equivalence, Root},
};

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
pub fn sort_to_bins<T: Ord>(sorted_keys: &[T], bins: &[T]) -> Vec<usize> {
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
        if bin_start <= key && key < bin_end {
            *r += 1;
            count += 1;
        } else {
            // Move the bin forward until it fits. There will always be a fitting bin.
            loop {
                if let Some((rn, (bsn, ben))) = bin_iter.next() {
                    if bsn <= key && key < ben {
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

/// Redistribute an array via an all_to_all_varcount operation.
pub fn redistribute<T: Equivalence, C: CommunicatorCollectives>(
    arr: &[T],
    counts: &[i32],
    comm: &C,
) -> Vec<T> {
    assert_eq!(counts.len(), comm.size() as usize);

    // First send the counts around via an alltoall operation.

    let mut recv_counts = vec![0; counts.len()];

    comm.all_to_all_into(counts, &mut recv_counts);

    // We have the recv_counts. Allocate space and setup the partitions.

    let nelems = recv_counts.iter().sum::<i32>() as usize;

    let mut output = Vec::<T>::with_capacity(nelems);
    let out_buf: &mut [T] = unsafe { std::mem::transmute(output.spare_capacity_mut()) };

    let send_partition = Partition::new(arr, counts, displacements(counts));
    let mut recv_partition =
        PartitionMut::new(out_buf, &recv_counts[..], displacements(&recv_counts));

    comm.all_to_all_varcount_into(&send_partition, &mut recv_partition);

    unsafe { output.set_len(nelems) };

    output
}

/// Compute displacements from a vector of counts.
///
/// This is useful for global MPI varcount operations. Let
/// count [ 3, 4, 5]. Then the corresponding displacements are
// [0, 3, 7]. Note that the last element `5` is ignored.
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
    out_data: &[T],
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

    let send_partition = mpi::datatype::Partition::new(out_data, counts, send_displacements);
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

/// Receiev the scattered data from `root`.
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
