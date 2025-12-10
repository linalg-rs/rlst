//! An implementation of bucket sort

use std::cmp::Ordering;

use itertools::Itertools;
use mpi::traits::{CommunicatorCollectives, Equivalence};
use paste::paste;
use rand::{Rng, seq::IndexedRandom};

use crate::{
    TotalCmp,
    distributed_tools::{
        all_to_allv,
        array_tools::{gather_to_all, global_max, global_min},
        sort_to_bins,
    },
};

/// Choose OVERSAMPLING * nprocs splitters for bucket sort
pub const OVERSAMPLING: usize = 8;

/// A helper trait for parallel sorts to convert items into unique items
///
/// A unique item is created by adding information about the rank
/// and local index to the item.
///
/// The easiest way is to create a struct of the form
///
/// ```
/// pub struct UniqueItem {
///   value: f64,
///   rank: usize,
///   index: usize,
/// }
/// ```
/// Here, `f64` is just an example and should be replaced by the suitable type.
///
/// Note that the type also requires [TotalCmp] and [Equivalence] to be implemented
/// so that comparisons for parallel sorting are possible.
///
/// This trait is already implemented for most scalar data types.
pub trait AsUnique: TotalCmp + Clone + Copy {
    /// The unique output type
    type Unique: Copy + Clone + TotalCmp + Equivalence;

    /// Convert `self` into the unique type.
    ///
    /// Here, `rank` is the process rank and `index` is the local index on the rank.
    fn into_unique(value: Self, rank: usize, index: usize) -> Self::Unique;

    /// Convert a unique item back into the original type.
    fn from_unique(elem: Self::Unique) -> Self;
}

/// Implementation of a parallel bucket sort
///
/// This function performs a global sort across all processes in a communicator
/// of the distributed array `arr`.
///
/// The algorithm is a simple bucket sort. It works as follows.
///
/// - Each item is converted into a unique type that stores also the rank
///   and local index. This is to ensure that each item in the global array
///   to be sorted appears only ones.
/// - [OVERSAMPLING] samples are chosen on each process and sent across all processes.
/// - Each process now splits up the samples into `nprocs` buckets, where `nprocs` is the number
///   of processes.
/// - Each process sends their local data to the process with the corresponding bucket.
/// - At the end each bucket is then locally sorted.
///
/// As part of the algorithm each process has to undertake two local sorts
/// and an `all_to_allv` communication across all processes.
///
/// If the MPI size is 1 the elements are sorted locally using `slice::sort_by`.
///
/// **The implementation of `parsort` is stable.**
///
/// It is only possible to sort types that implement the `AsUnique` trait which converts
/// elements into a unique representation. This is predefined for a number of primitive types:
/// `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `isize`, `u8`, `u16`, `u32`, `u64`, `usize`.
pub fn parsort<Item: AsUnique, C: CommunicatorCollectives, R: Rng + ?Sized>(
    arr: &[Item],
    comm: &C,
    rng: &mut R,
) -> Vec<Item> {
    let size = comm.size() as usize;
    let rank = comm.rank() as usize;
    // If we only have a single rank simply sort the local array and return

    let mut arr = arr.to_vec();

    if size == 1 {
        arr.sort_by(|x, y| x.total_cmp(*y));
        return arr;
    }

    // We first convert the array into unique elements by adding information
    // about index and rank. This guarantees that we don't have duplicates in
    // our sorting set.

    let mut arr = arr
        .iter()
        .enumerate()
        .map(|(index, elem)| AsUnique::into_unique(*elem, rank, index))
        .collect_vec();

    // We now sort the local array.

    arr.sort_by(|x, y| x.total_cmp(*y));

    // Let us now get the buckets.

    let bins = get_bins::<Item, _, _>(&arr, comm, rng);

    // We now redistribute with respect to these buckets.

    let counts = sort_to_bins(&arr, &bins);
    let mut recvbuffer = all_to_allv(comm, &counts, &arr).1;

    // We now have everything in the receive buffer. Now sort the local elements and return

    recvbuffer.sort_by(|x, y| x.total_cmp(*y));

    recvbuffer
        .iter()
        .map(|&elem| AsUnique::from_unique(elem))
        .collect_vec()
}

fn get_bins<Item, C, R>(
    arr: &[<Item as AsUnique>::Unique],
    comm: &C,
    rng: &mut R,
) -> Vec<<Item as AsUnique>::Unique>
where
    Item: AsUnique,
    C: CommunicatorCollectives,
    R: Rng + ?Sized,
{
    let size = comm.size() as usize;

    // In the first step we pick `oversampling * nprocs` splitters.

    let oversampling = if arr.len() < OVERSAMPLING {
        arr.len()
    } else {
        OVERSAMPLING
    };

    // We get the global smallest and global largest element. We do not want those
    // in the splitter so filter out their occurence.

    let global_min_elem = global_min(arr, comm);
    let global_max_elem = global_max(arr, comm);

    let splitters = arr
        .choose_multiple(rng, oversampling)
        .copied()
        .collect::<Vec<_>>();

    // We gather the splitters into all ranks so that each rank has all splitters.

    let mut all_splitters = gather_to_all(&splitters, comm);

    // We now have all splitters available on each process.
    // We can now sort the splitters. Every process will then have the same list of sorted splitters.

    all_splitters.sort_unstable_by(|&x, &y| x.total_cmp(y));

    // We now insert the smallest and largest possible element if they are not already
    // in the splitter collection.

    if !matches!(
        TotalCmp::total_cmp(*all_splitters.first().unwrap(), global_min_elem),
        Ordering::Equal,
    ) {
        all_splitters.insert(0, global_min_elem);
    }

    if !matches!(
        TotalCmp::total_cmp(*all_splitters.last().unwrap(), global_max_elem),
        Ordering::Equal,
    ) {
        all_splitters.push(global_max_elem);
    }

    // We now define p buckets (p is number of processors) and we return
    // a p element array containing the first element of each bucket

    all_splitters = split(&all_splitters, size)
        .map(|slice| slice.first().unwrap())
        .copied()
        .collect::<Vec<_>>();

    all_splitters
}

// The following is a simple iterator that splits a slice into n
// chunks. It is from https://users.rust-lang.org/t/how-to-split-a-slice-into-n-chunks/40008/3

fn split<T>(slice: &[T], n: usize) -> impl Iterator<Item = &[T]> {
    let len = slice.len() / n;
    let rem = slice.len() % n;
    Split { slice, len, rem }
}

struct Split<'a, T> {
    slice: &'a [T],
    len: usize,
    rem: usize,
}

impl<'a, T> Iterator for Split<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            return None;
        }
        let mut len = self.len;
        if self.rem > 0 {
            len += 1;
            self.rem -= 1;
        }
        let (chunk, rest) = self.slice.split_at(len);
        self.slice = rest;
        Some(chunk)
    }
}

macro_rules! impl_as_unique_for {
    ($dtype:ty) => {
        paste! {

             #[derive(Copy, Clone, PartialOrd, PartialEq, Equivalence)]
             #[allow(non_camel_case_types, missing_docs)]
             pub struct [<UniqueItem_ $dtype>] {
                 /// Stored value
                 pub value: $dtype,
                 /// Process rank
                 pub rank: usize,
                 /// Local index on process
                 pub index: usize,
             }

             impl TotalCmp for [<UniqueItem_$dtype>] {
                 fn total_cmp(self, other: Self) -> Ordering {
                     let ordering = self.value.total_cmp(other.value);
                     if !matches!(ordering, Ordering::Equal) {
                         ordering
                     } else {
                         let ordering = self.rank.cmp(&other.rank);

                         if !matches!(ordering, Ordering::Equal) {
                             ordering
                         } else {
                             self.index.cmp(&other.index)
                         }

                     }
                 }

             }

             impl AsUnique for $dtype {
                 type Unique = [<UniqueItem_$dtype>];

                 fn into_unique(value: $dtype, rank: usize, index: usize) -> Self::Unique {
                     [<UniqueItem_$dtype>] {value, rank, index}
                 }

                 fn from_unique(elem: Self::Unique) -> Self {
                     elem.value
                 }
             }

        } // End of paste
    };
}

impl_as_unique_for!(f64);
impl_as_unique_for!(f32);
impl_as_unique_for!(i8);
impl_as_unique_for!(i16);
impl_as_unique_for!(i32);
impl_as_unique_for!(i64);
impl_as_unique_for!(isize);
impl_as_unique_for!(u8);
impl_as_unique_for!(u16);
impl_as_unique_for!(u32);
impl_as_unique_for!(u64);
impl_as_unique_for!(usize);
