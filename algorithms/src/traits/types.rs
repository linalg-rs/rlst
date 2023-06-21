//! Types for linear algebra traits.

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum TriangularType {
    Upper = b'U',
    Lower = b'L',
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum TriangularDiagonal {
    Unit = b'U',
    NonUnit = b'N',
}

/// QR Mode
///
/// Full: Return the full Q matrix.
/// Reduced: Return the reduced Q matrix.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub enum QrMode {
    Full,
    Reduced,
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub enum PivotMode {
    NoPivoting,
    WithPivoting,
}

/// Transposition mode for Lapack.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum TransposeMode {
    /// No transpose
    NoTrans = b'N',
    /// Transpose
    Trans = b'T',
    /// Conjugate Transpose
    ConjugateTrans = b'C',
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum SideMode {
    Left = b'L',
    Right = b'R',
}
