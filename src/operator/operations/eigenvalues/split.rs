//! Module for splitting eigenvalue problems.
use crate::RlstScalar;

/// Slice working vector accordingly
pub fn xslice_yslice<'a, Item: RlstScalar>(
    workd: &'a mut [Item],
    n0: i32,
    n1: i32,
    n: i32,
) -> (&'a mut [Item], &'a mut [Item]) {
    let n = n as usize;
    if n1 - n0 > n as i32 {
        // n0,n2,n1
        let (xslice, tmp) = workd.split_at_mut(n);
        let (_, tmp) = tmp.split_at_mut(n);
        let (yslice, _) = tmp.split_at_mut(n);
        return (xslice, yslice);
    } else if n1 > n0 {
        // [n2,]n0,n1[,n2]
        let (_, tmp) = workd.split_at_mut(n0 as usize);
        let (xslice, tmp) = tmp.split_at_mut(n);
        let (yslice, _) = tmp.split_at_mut(n);
        return (xslice, yslice);
    } else if n0 - n1 > n as i32 {
        // n1,n2,n0
        let (yslice, tmp) = workd.split_at_mut(n);
        let (_, tmp) = tmp.split_at_mut(n);
        let (xslice, _) = tmp.split_at_mut(n);
        return (xslice, yslice);
    } else {
        // [n2,]n1,n0[,n2]
        let (_, tmp) = workd.split_at_mut(n1 as usize);
        let (yslice, tmp) = tmp.split_at_mut(n);
        let (xslice, _) = tmp.split_at_mut(n);
        return (xslice, yslice);
    }
}
