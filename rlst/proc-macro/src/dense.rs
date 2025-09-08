use proc_macro::TokenStream;
use quote::quote;
use syn::Lit;

pub(crate) fn rlst_static_array_impl(items: TokenStream) -> TokenStream {
    let args = items.to_string();
    let args: Vec<&str> = args.splitn(2, ',').map(|elem| elem.trim()).collect();
    let ty = args[0];
    let ty = if let Ok(ty) = syn::parse_str::<syn::Type>(ty) {
        ty
    } else {
        panic!("Type expected as first argument.");
    };
    let dims = if let Ok(syn::Expr::Array(expr_array)) = syn::parse_str(args[1]) {
        expr_array
    } else {
        panic!("Array of dimensions expected as second argument.");
    };

    let elems = dims.elems;
    let ndim: usize = elems.len();
    let dims = elems
        .iter()
        .map(|elem| {
            if let syn::Expr::Lit(elem) = elem {
                match &elem.lit {
                    Lit::Int(int_lit) => int_lit.base10_parse::<usize>().unwrap(),
                    _ => {
                        panic!("Integer expected as dimensional argument.");
                    }
                }
            } else {
                panic!("Not a literal expression");
            }
        })
        .collect::<Vec<usize>>();

    let output = quote! { {
        let data = rlst::dense::data_container::ArrayContainer::<#ty, #ndim>::new();
        rlst::dense::array::Array::new(rlst::dense::base_array::BaseArray::new(data, [#(#dims),*]))
    }
    };

    output.into()
}

pub(crate) fn rlst_dynamic_array_impl(items: TokenStream) -> TokenStream {
    let args = items.to_string();
    let args: Vec<&str> = args.splitn(2, ',').map(|elem| elem.trim()).collect();
    let ty = args[0];
    let ty = if let Ok(ty) = syn::parse_str::<syn::Type>(ty) {
        ty
    } else {
        panic!("Type expected as first argument.");
    };
    let dims = if let Ok(syn::Expr::Array(expr_array)) = syn::parse_str(args[1]) {
        expr_array
    } else {
        panic!("Array of dimensions expected as second argument.");
    };

    let ndims = dims.elems.len();

    let output = quote! {
        rlst::dense::array::DynArray::<#ty, #ndims>::from_shape(#dims)
    };

    output.into()
}

pub(crate) fn rlst_static_type_impl(items: TokenStream) -> TokenStream {
    let args = items.to_string();
    let args: Vec<&str> = args.split(',').map(|elem| elem.trim()).collect();
    let ty = args[0];
    let ty = if let Ok(syn::Type::Path(type_path)) = syn::parse_str(ty) {
        type_path
    } else {
        panic!("Type expected as first argument.");
    };
    let dims = args[1..]
        .iter()
        .map(|elem| elem.parse::<usize>().unwrap())
        .collect::<Vec<usize>>();

    let nelem: usize = dims.iter().product();
    let ndim = args[1..].len();

    let output = quote! { rlst::dense::array::Array<#ty, rlst::dense::base_array::BaseArray<#ty, rlst::dense::data_container::ArrayContainer<#ty, #nelem>, #ndim>, #ndim>
    };

    output.into()
}
