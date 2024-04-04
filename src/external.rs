//! Interfaces to external libraries

// Only include Metal on Macos running on Apple Silicon
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod metal;
