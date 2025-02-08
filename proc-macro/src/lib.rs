//! Proc Macros for the rlst library
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

use proc_macro::TokenStream;

mod dense;
mod tracing;

/// Create a static array
#[proc_macro]
pub fn rlst_static_array(items: TokenStream) -> TokenStream {
    dense::rlst_static_array_impl(items)
}

/// Type of a static array
#[proc_macro]
pub fn rlst_static_type(items: TokenStream) -> TokenStream {
    dense::rlst_static_type_impl(items)
}

/// Trace the execution time of a function
#[proc_macro_attribute]
pub fn measure_duration(attr: TokenStream, item: TokenStream) -> TokenStream {
    tracing::measure_duration_impl(attr, item)
}
