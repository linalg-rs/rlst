//! Traits for I/O Operations.

/// Identifier trait to associate a matrix market identifier with a Rust type.
pub trait MmIdentifier {
    /// Matrix market type
    const MMTYPE: &'static str;
}

impl MmIdentifier for f32 {
    const MMTYPE: &'static str = "real";
}

impl MmIdentifier for f64 {
    const MMTYPE: &'static str = "real";
}

impl MmIdentifier for crate::base_types::c32 {
    const MMTYPE: &'static str = "complex";
}

impl MmIdentifier for crate::base_types::c64 {
    const MMTYPE: &'static str = "complex";
}
