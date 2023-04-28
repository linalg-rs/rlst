//! Dimension trait

pub trait Shape {
    fn shape(&self) -> (usize, usize);
}

pub trait NumberOfElements {
    fn number_of_elements(&self) -> usize;
}
