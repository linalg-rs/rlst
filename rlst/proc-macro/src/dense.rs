use proc_macro::TokenStream;
use quote::quote;

pub(crate) fn rlst_static_array_impl(items: TokenStream) -> TokenStream {
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

    let ndim: usize = dims.iter().product();

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
    let ty = if let Ok(syn::Type::Path(type_path)) = syn::parse_str(ty) {
        type_path
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
