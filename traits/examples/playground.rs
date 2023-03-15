pub use rlst_traits::*;
use std::fmt::Debug;
use std::marker::PhantomData;

// We create two structs. One will have a matvec,
// the other one not.
#[derive(Debug)]
struct OpWithMatVec {}
#[derive(Debug)]
struct OpWithoutMatVec {}

struct SimpleSpace;

impl LinearSpace for SimpleSpace {
    type F = f64;
    type E<'a> = Vec;
}

struct View<'a> {
    marker: PhantomData<&'a ()>,
}

// Simple helper structs as mock vectors
#[derive(Debug)]
struct Vec {}

impl<'a> View<'a> {
    fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl Element for Vec {
    type Space = SimpleSpace;
    type View<'a> = View<'a> where Self: 'a;
    type ViewMut<'a> = View<'a> where Self: 'a;

    fn view<'a>(&'a self) -> Self::View<'a> {
        View::new()
    }

    fn view_mut<'a>(&'a mut self) -> Self::View<'a> {
        View::new()
    }
}

// For OpWithMatvec we have to implement
// `as_matvec` to return a reference to self.
// This is just boilerplate and could be handled
// by a simple derive macro or similar so the user
// needs not write this boilerplate.
impl OperatorBase for OpWithMatVec {
    type Domain = SimpleSpace;
    type Range = SimpleSpace;
    fn as_apply(&self) -> Option<&dyn AsApply<Domain = Self::Domain, Range = Self::Range>> {
        Some(self)
    }
}

// The actual matvec is now implemented. It is just
// a stub that prints a message.
impl AsApply for OpWithMatVec {
    fn apply(
        &self,
        _x: ElementView<Self::Domain>,
        _y: ElementViewMut<Self::Range>,
    ) -> SparseLinAlgResult<()> {
        println!("I am doing a matvec");
        Ok(())
    }
}

// For the operator without matvec we need no boilerplate
// It just needs to implement `OperatorBase` as empty trait.
impl OperatorBase for OpWithoutMatVec {
    type Domain = SimpleSpace;
    type Range = SimpleSpace;
}

fn main() {
    // We create two structs. One witho matvec and one without
    let op_with_matvec = OpWithMatVec {};
    //let op_matvec_ref = &op_with_matvec as &dyn OperatorBase;

    let op_without_matvec = OpWithoutMatVec {};

    let x = Vec {};
    let mut y = Vec {};

    // In the following code the cast to the base trait object is only
    // done to emphasise that we do not need to operate on the concrete
    // type. But the vtable of OperatorBase has everything we need.

    // For op_with_matvec it executes the matvec.
    if let Some(obj) =
        (&op_with_matvec as &dyn OperatorBase<Domain = SimpleSpace, Range = SimpleSpace>).as_apply()
    {
        obj.apply(x.view(), y.view_mut()).unwrap();
    } else {
        // It never goes into this if branch
        println!("Cannot find matvec for op_with_matvec.");
    }

    // For op_without_matvec it does not execute the matvec.
    if let Some(obj) = (&op_without_matvec
        as &dyn OperatorBase<Domain = SimpleSpace, Range = SimpleSpace>)
        .as_apply()
    {
        obj.apply(x.view(), y.view_mut()).unwrap();
    } else {
        // It always goes into this branch.
        println!("Cannot find matvec for op_without_matvec.");
    }

    // Let's now test the has_matvec routine. Again we cast
    // to the base trait object just to demonstrate that we only
    // need the base trait.

    println!(
        "Does op_with_matvec support matvec? {:#?}",
        (&op_with_matvec as &dyn OperatorBase<Domain = SimpleSpace, Range = SimpleSpace>)
            .has_apply()
    );
    println!(
        "Does op_without_matvec support matvec? {:#?}",
        (&op_without_matvec as &dyn OperatorBase<Domain = SimpleSpace, Range = SimpleSpace>)
            .has_apply()
    );
}
