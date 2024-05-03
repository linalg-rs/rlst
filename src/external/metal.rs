//! Interface to Apple Metal.

pub mod interface;
pub mod linalg;
pub mod metal_array;
pub mod raw;

pub use interface::AutoReleasePool;
pub use interface::MetalDevice;
pub use metal_array::AsRawMetalBuffer;
pub use metal_array::MetalDataBuffer;
