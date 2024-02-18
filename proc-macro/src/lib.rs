//! Proc Macros for the rlst library

use proc_macro::TokenStream;

mod dense;

#[proc_macro_attribute]
pub fn rlst_static_size(args: TokenStream, input: TokenStream) -> TokenStream {
    dense::rlst_static_size_impl(args, input)
}
#[proc_macro]
pub fn rlst_static_array(items: TokenStream) -> TokenStream {
    dense::rlst_static_array_impl(items)
}

#[proc_macro]
pub fn rlst_static_type(items: TokenStream) -> TokenStream {
    dense::rlst_static_type_impl(items)
}
