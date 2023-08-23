//! Proc Macros for the rlst library

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, ItemStruct};

#[proc_macro_attribute]
pub fn rlst_fixed_size(args: TokenStream, input: TokenStream) -> TokenStream {
    let input_cloned = input.clone();
    let ast = parse_macro_input!(input_cloned as ItemStruct);
    let name = ast.ident;

    let args = args.to_string();

    println!("Name: {}", name);
    println!("Args: {}", args);

    let dims: Vec<&str> = args.split(',').map(|elem| elem.trim()).collect();
    let m = dims[0].parse::<usize>().unwrap();
    let n = dims[1].parse::<usize>().unwrap();
    println!("{}", input);

    println!("Dimension: ({}, {})", m, n);

    let gen = quote! {
        #input
    };
    gen.into()
    //input
}

fn impl_rlst_fixed_size_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let gen = quote! {
            impl rlst_dense::traits::ExperimentalSizeIdentifier for #name  {
                const SIZE: rlst_dense::traits::SizeValue = rlst_dense::traits::SizeValue::Fixed(3, 4);
            }
    };
    gen.into()
}
