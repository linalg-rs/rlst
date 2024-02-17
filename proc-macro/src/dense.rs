use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemStruct};

pub(crate) fn rlst_static_size_impl(args: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemStruct);
    let struct_name = input.ident.clone();

    let args = args.to_string();

    let dims: Vec<&str> = args.split(',').map(|elem| elem.trim()).collect();
    let m = dims[0].parse::<usize>().unwrap();
    let n = dims[1].parse::<usize>().unwrap();

    let output = quote! {
        #input

        impl rlst_dense::traits::SizeIdentifier for #struct_name {
            const SIZE: rlst_dense::traits::SizeIdentifierValue = rlst_dense::traits::SizeIdentifierValue::Static(#m, #n);
        }

        impl<T: rlst_common::types::Scalar> rlst_dense::traits::MatrixBuilder<T> for #struct_name {
            type Out = rlst_dense::GenericBaseMatrix::<T, rlst_dense::ArrayContainer<T, {#m * #n}>, #struct_name>;


            fn new_matrix(dim: (usize, usize)) -> Self::Out {

            assert_eq!(dim, (#m, #n), "Expected fixed dimension ({}, {}) for static matrix.", #m, #n);
            use rlst_dense::LayoutType;
            <Self::Out>::from_data(rlst_dense::ArrayContainer::<T, {#m * #n}>::new(),
            rlst_dense::DefaultLayout::from_dimension((#m, #n)),
        )

            }
        }

    };
    output.into()
}

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
        let data = rlst_dense::data_container::ArrayContainer::<#ty, #ndim>::new();
        rlst_dense::array::Array::new(rlst_dense::base_array::BaseArray::new(data, [#(#dims),*]))
    }
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

    let output = quote! { rlst_dense::array::Array<#ty, rlst_dense::base_array::BaseArray<#ty, rlst_dense::data_container::ArrayContainer<#ty, #nelem>, #ndim>, #ndim>
    };

    output.into()
}
