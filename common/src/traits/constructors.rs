pub trait NewFromZero {
    fn new_from_zero(&self) -> Self;
}

pub trait Duplicate {
    // Duplicate an object

    fn duplicate(&self) -> Self;
}
