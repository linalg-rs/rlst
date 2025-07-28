//! Binary operations on sparse matrices.

use std::iter::Peekable;

/// A binary operator that takes two iterators over sparse matrix entries and applies a function to them.
pub struct BinaryAijOperator<Item1, Item2, Iter1, Iter2, Out, F>
where
    Item1: Copy,
    Item2: Copy,
    Out: Copy,
    F: Fn(Item1, Item2) -> Out,
    Iter1: Iterator<Item = ([usize; 2], Item1)>,
    Iter2: Iterator<Item = ([usize; 2], Item2)>,
{
    iter1: Peekable<Iter1>,
    iter2: Peekable<Iter2>,
    func: F,
    finished: bool,
}

impl<Item1, Item2, Iter1, Iter2, Out, F> BinaryAijOperator<Item1, Item2, Iter1, Iter2, Out, F>
where
    Item1: Copy,
    Item2: Copy,
    Out: Copy,
    F: Fn(Item1, Item2) -> Out,
    Iter1: Iterator<Item = ([usize; 2], Item1)>,
    Iter2: Iterator<Item = ([usize; 2], Item2)>,
{
    /// Create a new binary operator.
    pub fn new(iter1: Iter1, iter2: Iter2, func: F) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
            func,
            finished: false,
        }
    }
}

impl<Item1, Item2, Iter1, Iter2, Out, F> Iterator
    for BinaryAijOperator<Item1, Item2, Iter1, Iter2, Out, F>
where
    Item1: Copy + Default,
    Item2: Copy + Default,
    Out: Copy + Default + PartialEq,
    F: Fn(Item1, Item2) -> Out,
    Iter1: Iterator<Item = ([usize; 2], Item1)>,
    Iter2: Iterator<Item = ([usize; 2], Item2)>,
{
    type Item = ([usize; 2], Out);

    fn next(&mut self) -> Option<Self::Item> {
        // Helper function to compare two indices.
        fn compare(a1: [usize; 2], a2: [usize; 2]) -> std::cmp::Ordering {
            if a1[0] < a2[0] {
                std::cmp::Ordering::Less
            } else if a1[0] > a2[0] {
                std::cmp::Ordering::Greater
            } else if a1[1] < a2[1] {
                std::cmp::Ordering::Less
            } else if a1[1] > a2[1] {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        }

        // We peek both iterators and then take the actual smaller one.
        // If both are identical we combine them. Otherwise we combine the smaller one with a zero
        // on the other side.
        // If both iterators are None we also return None.

        // Easy case. The iterator is finished.
        if self.finished {
            return None;
        }

        let next1 = self.iter1.peek();
        let next2 = self.iter2.peek();

        // We compute the next potential value. However, before we return it we need to make sure
        // that it did not compute a zero value. If that's the case we need to skip it.
        let out_candidate = if next1.is_none() && next2.is_none() {
            // If both iterators are None we are done
            // We only peeked, so actually advance them now to finish them.
            self.iter1.next();
            self.iter2.next();
            self.finished = true;
            None
        } else if next1.is_some() && next2.is_none() {
            // Only the first iterator has a value.
            self.iter2.next();
            let (index, value) = self.iter1.next().unwrap();
            Some((index, (self.func)(value, Item2::default())))
        } else if next1.is_none() && next2.is_some() {
            // Only the second iterator has a value.
            self.iter1.next(); // Still advance iter1 as it was only peeked.
            let (index, value) = self.iter2.next().unwrap();
            Some((index, (self.func)(Item1::default(), value)))
        } else {
            // Both iterators have a value.
            let (index1, value1) = *next1.unwrap();
            let (index2, value2) = *next2.unwrap();

            match compare(index1, index2) {
                std::cmp::Ordering::Less => {
                    // The first iterator has the smaller index.
                    // We only advance this.
                    self.iter1.next(); // Advance iter1 as it was only peeked.
                    Some((index1, (self.func)(value1, Item2::default())))
                }
                std::cmp::Ordering::Greater => {
                    // The second iterator has the smaller index.
                    // We only advance this.
                    self.iter2.next(); // Advance iter2 as it was only peeked.
                    Some((index2, (self.func)(Item1::default(), value2)))
                }
                std::cmp::Ordering::Equal => {
                    // Both iterators have the same index.
                    self.iter1.next(); // Advance both iterators.
                    self.iter2.next();
                    Some((index1, (self.func)(value1, value2)))
                }
            }
        };

        // Investigate that out_candidate was not a zero value.
        // We do not want to return zero values for sparse matrices.

        if let Some((index, out_value)) = out_candidate {
            if out_value != Out::default() {
                // If the value is not zero we return it.
                Some((index, out_value))
            } else {
                // If the value is zero we continue to the next iteration.
                self.next()
            }
        } else {
            // If there was no candidate we return None.
            None
        }
    }
}
