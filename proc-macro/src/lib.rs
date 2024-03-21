//! Proc Macros for the rlst library
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

use proc_macro::TokenStream;

mod dense;

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
