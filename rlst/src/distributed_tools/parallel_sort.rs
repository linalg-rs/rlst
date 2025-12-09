//! An implementation of bucket sort

use std::{cmp::Ordering, fmt::Display};

use itertools::Itertools;
use mpi::traits::{CommunicatorCollectives, Equivalence};
use rand::{Rng, seq::IndexedRandom};

use crate::{
    Max, Min,
    distributed_tools::{
        all_to_allv,
        array_tools::{gather_to_all, global_max, global_min},
        sort_to_bins,
    },
};

/// Choose OVERSAMPLING * nprocs splitters for bucket sort
const OVERSAMPLING: usize = 8;

pub type SortType = f64;

/// An internal struct. We convert every array element
/// into this struct. The idea is that this is guaranteed to be unique
/// as it encodes not only the element but also its rank and index.
#[derive(Copy, Clone, Default, PartialOrd, PartialEq, Equivalence)]
struct UniqueItem {
    pub value: SortType,
    pub rank: usize,
    pub index: usize,
}

impl Min for UniqueItem {
    type Output = UniqueItem;

    fn min(self, other: Self) -> Self::Output {
        if self < other { self } else { other }
    }
}

impl Max for UniqueItem {
    type Output = UniqueItem;

    fn max(self, other: Self) -> Self::Output {
        if self > other { self } else { other }
    }
}

impl Display for UniqueItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(value: {}, rank: {}, index: {})",
            self.value, self.rank, self.index
        )
    }
}

impl UniqueItem {
    pub fn new(value: SortType, rank: usize, index: usize) -> Self {
        Self { value, rank, index }
    }
}

fn to_unique_item(arr: &[SortType], rank: usize) -> Vec<UniqueItem> {
    arr.iter()
        .enumerate()
        .map(|(index, &item)| UniqueItem::new(item, rank, index))
        .collect()
}

fn get_bins<C, R>(arr: &[UniqueItem], comm: &C, rng: &mut R) -> Vec<UniqueItem>
where
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

    all_splitters.sort_unstable_by(|x, y| {
        if *x < *y {
            Ordering::Less
        } else if *x == *y {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    });

    // We now insert the smallest and largest possible element if they are not already
    // in the splitter collection.

    if *all_splitters.first().unwrap() != global_min_elem {
        all_splitters.insert(0, global_min_elem)
    }

    if *all_splitters.last().unwrap() != global_max_elem {
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

/// Parallel sort
pub fn parsort<C: CommunicatorCollectives, R: Rng + ?Sized>(
    arr: &[SortType],
    comm: &C,
    rng: &mut R,
) -> Vec<SortType> {
    let size = comm.size() as usize;
    let rank = comm.rank() as usize;
    // If we only have a single rank simply sort the local array and return

    let mut arr = arr.to_vec();

    if size == 1 {
        arr.sort_unstable_by(|x, y| {
            if *x < *y {
                Ordering::Less
            } else if *x == *y {
                Ordering::Equal
            } else {
                Ordering::Greater
            }
        });
        return arr;
    }

    // We first convert the array into unique elements by adding information
    // about index and rank. This guarantees that we don't have duplicates in
    // our sorting set.

    let mut arr = to_unique_item(&arr, rank);

    // We now sort the local array.

    arr.sort_unstable_by(|x, y| {
        if *x < *y {
            Ordering::Less
        } else if *x == *y {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    });

    // Let us now get the buckets.

    let bins = get_bins(&arr, comm, rng);

    // We now redistribute with respect to these buckets.

    let counts = sort_to_bins(&arr, &bins);
    let mut recvbuffer = all_to_allv(comm, &counts, &arr).1;

    // We now have everything in the receive buffer. Now sort the local elements and return

    recvbuffer.sort_unstable_by(|x, y| {
        if *x < *y {
            Ordering::Less
        } else if *x == *y {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    });
    recvbuffer.iter().map(|&elem| elem.value).collect_vec()
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
