//! Definition of distributed CSR matrices.
//!
//! A distributed CSR matrix `A` is a matrix whose rows are distributed across processes. A distributed
//! CSR matrix requires two index layouts, a domain layout and a range layout. The domain layout specifies
//! how a vector `x` is distributed for the matrix-vector product `Ax`. The range layout specifies how the rows
//! of `A` are distributed.
//!
//! A distributed CSR matrix is locally on each process simply a standard CSR matrix. Upon definition ghost dofs
//! on each process are identified from the domain layout and the domain dofs reordered such that the local dofs
//! are ordered before the ghost dofs. This happens transparently to the user and uppon applying a matrix-vector product
//! the dofs are automatically sorted in the right way.
//!
//! A distributed CSR matrix is easiest instantiated through the [DistributedCsrMatrix::from_aij] function.
//! Every process simply contributes matrix elements and their indices. These matrix elements need not belong
//! to the local process. The function sends all elements to their correct processes, then sums up duplicates
//! and instantiates the local CSR matrices in the correct way.
//!
use std::collections::HashMap;
use std::ops::{Add, AddAssign, Mul};
use std::rc::Rc;

use itertools::{izip, Itertools};
use mpi::collective::CommunicatorCollectives;
use mpi::traits::{Communicator, Equivalence};
use num::One;

use crate::dense::array::DynArray;
use crate::distributed_tools::{redistribute, sort_to_bins, GhostCommunicator, IndexLayout};
use crate::{
    empty_array, AijIteratorByValue, AsMatrixApply, BaseItem, FromAijDistributed, Nonzeros, Shape,
    SparseMatrixType, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

use super::csr_mat::CsrMatrix;
use super::distributed_array::DistributedArray;
use super::tools::normalize_aij;
use super::SparseMatType;

/// Distributed CSR matrix
pub struct DistributedCsrMatrix<'a, Item, C>
where
    C: Communicator,
{
    /// The type of the sparse matrix.
    mat_type: SparseMatType,
    /// The local matrix.
    local_matrix: CsrMatrix<Item>,
    /// The local number of dofs.
    local_dof_count: usize,
    /// The domain layout.
    domain_layout: Rc<IndexLayout<'a, C>>,
    range_layout: Rc<IndexLayout<'a, C>>,
    domain_ghosts: GhostCommunicator<usize>,
    local2global: Vec<usize>,
    global2local: HashMap<usize, usize>,
}

impl<'a, Item, C: Communicator> DistributedCsrMatrix<'a, Item, C>
where
    Item: Copy,
    C: Communicator,
{
    /// Create a new distributed CSR matrix. Users should call [DistributedCsrMatrix::from_aij] instead.
    pub fn new(
        indices: DynArray<usize, 1>,
        indptr: DynArray<usize, 1>,
        data: DynArray<Item, 1>,
        domain_layout: Rc<IndexLayout<'a, C>>,
        range_layout: Rc<IndexLayout<'a, C>>,
    ) -> Self {
        // Both layouts must have the same communicator.
        assert!(std::ptr::addr_eq(domain_layout.comm(), range_layout.comm()));

        // The indptr vector must have one more element than the number of local indices.
        assert_eq!(1 + range_layout.number_of_local_indices(), indptr.len());

        let comm = domain_layout.comm();

        let my_rank = comm.rank() as usize;

        let domain_ghost_dofs: Vec<usize> = indices
            .iter_value()
            .unique()
            .filter(|&dof| domain_layout.rank_from_index(dof).unwrap() != my_rank)
            .collect();

        let ranks = domain_ghost_dofs
            .iter()
            .map(|dof| domain_layout.rank_from_index(*dof).unwrap())
            .collect::<Vec<_>>();

        let domain_ghosts = GhostCommunicator::new(&domain_ghost_dofs, &ranks, comm);
        let local_dof_count =
            domain_layout.number_of_local_indices() + domain_ghosts.total_receive_count;
        let mut global2local = HashMap::new();

        // We need to transform the indices vector of the CSR matrix from global indexing to
        // local indexing. For this we assume that the input vector has the format
        // [local_indices..., ghost_indices]. So we map global indices to fit this format.
        // To do this we create a hash map that takes the global indices and maps to the new
        // local indexing.

        let mut count: usize = 0;
        for index in domain_layout.local_range().0..domain_layout.local_range().1 {
            global2local.insert(index, count);
            count += 1;
        }
        for index in &domain_ghosts.receive_indices {
            global2local.insert(*index, count);
            count += 1;
        }

        // The hash map is created. We now iterate through the indices vector of the sparse matrix
        // to change the indexing.

        let mapped_indices = indices
            .iter_value()
            .map(|elem| *global2local.get(&elem).unwrap())
            .collect::<Vec<_>>();

        // Now reverese the  index mapper to get the local to global mapping.

        let local2global = global2local
            .iter()
            .sorted_by_key(|&(_, &local_index)| local_index)
            .map(|(&global_index, _)| global_index)
            .collect::<Vec<_>>();

        Self {
            mat_type: SparseMatType::DistCsr,
            local_matrix: CsrMatrix::new(
                [range_layout.number_of_local_indices(), count],
                DynArray::from_shape_and_vec([mapped_indices.len()], mapped_indices),
                indptr,
                data,
            ),
            local_dof_count,
            domain_layout,
            range_layout,
            domain_ghosts,
            local2global,
            global2local,
        }
    }

    /// Return the local sparse matrix
    pub fn local(&self) -> &CsrMatrix<Item> {
        &self.local_matrix
    }

    /// Return a mutable reference to the local sparse matrix
    pub fn local_mut(&mut self) -> &mut CsrMatrix<Item> {
        &mut self.local_matrix
    }

    /// Domain layout
    pub fn domain_layout(&self) -> Rc<IndexLayout<'a, C>> {
        self.domain_layout.clone()
    }

    /// Range layout
    pub fn range_layout(&self) -> Rc<IndexLayout<'a, C>> {
        self.range_layout.clone()
    }

    /// Communicator
    pub fn comm(&self) -> &C {
        self.domain_layout.comm()
    }

    ///  Map a global domain index to a domain index in the local sparse matrix.
    /// Return None if the index does not exist.
    pub fn domain_global2local(&self, global_index: usize) -> Option<usize> {
        self.global2local.get(&global_index).copied()
    }

    /// Map a local domain index to a  global domain index.
    pub fn domain_local2global(&self, local_index: usize) -> Option<usize> {
        self.local2global.get(local_index).copied()
    }

    /// Map a global range index to a range index in the local sparse matrix.
    pub fn range_global2local(&self, global_index: usize) -> Option<usize> {
        let offset = self.range_layout.local_range().0;
        if global_index < offset
            || global_index >= offset + self.range_layout.number_of_local_indices()
        {
            None
        } else {
            Some(global_index - offset)
        }
    }

    ///  Map a local range index to a global range index.
    pub fn range_local2global(&self, local_index: usize) -> Option<usize> {
        let offset = self.range_layout.local_range().0;
        if local_index < self.range_layout.number_of_local_indices() {
            Some(local_index + offset)
        } else {
            None
        }
    }
}

impl<'a, Item, C: Communicator> Nonzeros for DistributedCsrMatrix<'a, Item, C>
where
    CsrMatrix<Item>: Nonzeros,
    Item: Copy,
{
    fn nnz(&self) -> usize {
        let local_result = self.local_matrix.nnz();
        let mut global_result = 0;

        self.comm().all_reduce_into(
            &local_result,
            &mut global_result,
            mpi::collective::SystemOperation::sum(),
        );

        global_result
    }
}

impl<'a, Item: Copy + Default + Equivalence, C: Communicator> BaseItem
    for DistributedCsrMatrix<'a, Item, C>
{
    type Item = Item;
}

impl<'a, Item, C: Communicator> Shape<2> for DistributedCsrMatrix<'a, Item, C>
where
    CsrMatrix<Item>: Shape<2>,
{
    fn shape(&self) -> [usize; 2] {
        [
            self.range_layout.number_of_global_indices(),
            self.domain_layout.number_of_global_indices(),
        ]
    }
}

impl<'a, Item, C: Communicator> SparseMatrixType for DistributedCsrMatrix<'a, Item, C> {
    fn mat_type(&self) -> SparseMatType {
        self.mat_type
    }
}

impl<'a, Item, C> AijIteratorByValue for DistributedCsrMatrix<'a, Item, C>
where
    CsrMatrix<Item>: AijIteratorByValue<Item = Item>,
    Item: Copy + Default + Equivalence,
    C: Communicator,
{
    fn iter_aij_value(&self) -> impl Iterator<Item = ([usize; 2], Self::Item)> + '_ {
        self.local_matrix.iter_aij_value().map(|(index, value)| {
            let global_row = index[0] + self.range_layout().local_range().0;
            let global_col = self.local2global[index[1]];
            ([global_row, global_col], value)
        })
    }
}

impl<'a, Item, C> FromAijDistributed<'a> for DistributedCsrMatrix<'a, Item, C>
where
    Item: Default + Copy + AddAssign + PartialEq + Equivalence,
    C: Communicator,
{
    type C = C;

    fn from_aij(
        domain_layout: Rc<IndexLayout<'a, Self::C>>,
        range_layout: Rc<IndexLayout<'a, Self::C>>,
        rows: &[usize],
        cols: &[usize],
        data: &[<Self as crate::BaseItem>::Item],
    ) -> Self {
        // Require the communicators to be identical.

        assert!(std::ptr::addr_eq(domain_layout.comm(), range_layout.comm()));

        let comm = domain_layout.comm();

        let (rows, cols, data) = normalize_aij(rows, cols, data, SparseMatType::Csr);

        // We now exchange data across the processes that should not be on our node. We need to send to each node
        // the amount of data that it should get from us.
        // The data is already sorted by rows. So just need to iterate through and work out how much data each process
        // gets and the corresponding displacements.

        // First we create the index bounds

        let index_bounds = (0..comm.size())
            .map(|rank| range_layout.index_range(rank as usize).unwrap().0)
            .collect_vec();

        // Now we compute how many entries each process gets.
        let counts = sort_to_bins(&rows, &index_bounds)
            .iter()
            .map(|&x| x as i32)
            .collect_vec();

        let rows = redistribute(&rows, &counts, comm);
        let cols = redistribute(&cols, &counts, comm);
        let data = redistribute(&data, &counts, comm);

        // We now need to normalize again since processes could now again have elements at the same matrix positions being
        // sent over from different ranks.

        let (rows, cols, data) = normalize_aij(&rows, &cols, &data, SparseMatType::Csr);

        // We now have all the data at the right processes.
        // We can now create the indptr array.

        // First create the special case that there are no rows at our local process.

        if rows.is_empty() {
            // The index pointer of an empty sparse matrix still has one element.
            // It contains the total number of elements, namely zero.
            let indptr = empty_array::<_, 1>();
            let indices = empty_array::<_, 1>();
            let data = empty_array::<_, 1>();

            Self::new(indices, indptr, data, domain_layout, range_layout)
        } else {
            let mut indptr =
                Vec::<usize>::with_capacity(1 + range_layout.number_of_local_indices());
            let nelems = data.len();

            // The actual rows in the aij format start at a nonzero index
            // When we iterate through in the following loop we need to
            // take this into account.
            let first_row = range_layout.local_range().0;

            let mut count: usize = 0;
            for row in first_row..first_row + range_layout.number_of_local_indices() {
                indptr.push(count);
                while count < nelems && row == rows[count] {
                    count += 1;
                }
            }
            indptr.push(count);

            Self::new(
                cols.into(),
                indptr.into(),
                data.into(),
                domain_layout,
                range_layout,
            )
        }
    }
}

// impl<'a, Item, C, ArrayImplX, const NDIM: usize> DistributedCsrMatrix<'a, Item, C> {
//     pub fn dot(&self, other: &Array<ArrayImplX, NDIM>)
// }

impl<'a, Item, C, ArrayImplX, ArrayImplY>
    AsMatrixApply<DistributedArray<'a, C, ArrayImplX, 1>, DistributedArray<'a, C, ArrayImplY, 1>>
    for DistributedCsrMatrix<'a, Item, C>
where
    Item: Default
        + Mul<Output = Item>
        + AddAssign<Item>
        + Add<Output = Item>
        + Copy
        + One
        + Equivalence,
    C: Communicator,
    ArrayImplX: UnsafeRandom1DAccessByValue<Item = Item>
        + UnsafeRandomAccessByValue<1, Item = Item>
        + Shape<1>,
    ArrayImplY: UnsafeRandom1DAccessMut<Item = Item> + Shape<1>,
    // CsrMatrix<Item>: AsMatrixApply<DynArray<Item, 1>, Array<ArrayImplY, 1>, Item = Item>,
{
    fn apply(
        &self,
        alpha: Self::Item,
        x: &DistributedArray<'a, C, ArrayImplX, 1>,
        beta: Self::Item,
        y: &mut DistributedArray<'a, C, ArrayImplY, 1>,
    ) {
        assert!(self.domain_layout.is_same(&x.index_layout));
        assert!(self.range_layout.is_same(&y.index_layout));

        // Create a vector that combines local dofs and ghosts

        let my_rank = self.domain_layout.comm().rank() as usize;

        let out_values = {
            let mut out_values =
                Vec::<Self::Item>::with_capacity(self.domain_ghosts.total_send_count());
            let out_buff: &mut [Self::Item] =
                unsafe { std::mem::transmute(out_values.spare_capacity_mut()) };

            for (out, out_index) in
                izip!(out_buff.iter_mut(), self.domain_ghosts.send_indices.iter())
            {
                *out = x
                    .local
                    .get_value([self
                        .domain_layout()
                        .global2local(my_rank, *out_index)
                        .unwrap()])
                    .unwrap();
            }

            unsafe { out_values.set_len(self.domain_ghosts.total_send_count()) };
            out_values
        };

        let local_vec = {
            let mut local_vec = Vec::<Self::Item>::with_capacity(self.local_dof_count);
            local_vec.extend(x.local.iter_value());
            let ghost_data: &mut [Item] =
                unsafe { std::mem::transmute(local_vec.spare_capacity_mut()) };

            // Prepare the values that are sent to other ranks

            self.domain_ghosts
                .forward_send_values(&out_values, ghost_data);
            unsafe { local_vec.set_len(self.local_dof_count) };
            local_vec
        };

        // Compute result
        self.local_matrix
            .apply(alpha, &local_vec.into(), beta, &mut y.local);
    }
}

impl<'a, Item, C, ArrayImplX, ArrayImplY>
    AsMatrixApply<DistributedArray<'a, C, ArrayImplX, 2>, DistributedArray<'a, C, ArrayImplY, 2>>
    for DistributedCsrMatrix<'a, Item, C>
where
    Item: Default
        + Mul<Output = Item>
        + AddAssign<Item>
        + Add<Output = Item>
        + Copy
        + One
        + Equivalence,
    C: Communicator,
    ArrayImplX: UnsafeRandom1DAccessByValue<Item = Item>
        + UnsafeRandomAccessByValue<2, Item = Item>
        + Shape<2>,
    ArrayImplY: UnsafeRandom1DAccessMut<Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + Shape<2>
        + 'a,
{
    fn apply(
        &self,
        alpha: Self::Item,
        x: &DistributedArray<'a, C, ArrayImplX, 2>,
        beta: Self::Item,
        y: &mut DistributedArray<'a, C, ArrayImplY, 2>,
    ) {
        assert_eq!(
            x.shape()[1],
            y.shape()[1],
            "x and y have incompatible number of columns {} != {}",
            x.shape()[1],
            y.shape()[1]
        );

        for (colx, mut coly) in izip!(x.col_iter(), y.col_iter_mut()) {
            self.apply(alpha, &colx, beta, &mut coly);
        }
    }
}
