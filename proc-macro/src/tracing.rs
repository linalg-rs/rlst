//! Macro to trace execution times

use darling::{ast::NestedMeta, Error, FromMeta};
use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

#[derive(Default, FromMeta)]
#[darling(default)]
struct Identifier {
    id: String,
    stdout: bool,
}

pub(crate) fn measure_duration_impl(args: TokenStream, item: TokenStream) -> TokenStream {
    let syn::ItemFn {
        vis,
        sig,
        attrs,
        block,
        ..
    } = parse_macro_input!(item);

    let attr_args = match NestedMeta::parse_meta_list(args.into()) {
        Ok(v) => v,
        Err(e) => {
            return TokenStream::from(Error::from(e).write_errors());
        }
    };

    let args = match Identifier::from_list(&attr_args) {
        Ok(v) => v,
        Err(e) => {
            return TokenStream::from(e.write_errors());
        }
    };

    let ident = args.id;
    let stdout = args.stdout;

    let statements = block.stmts;

    if cfg!(feature = "enable_tracing") {
        quote! {


            # (# attrs)*
            #vis #sig
            {

                let now = std::time::Instant::now();

                let __result = {
                    #(#statements)*
                };

                let duration = now.elapsed();

                #[cfg(feature="mpi")]
                if #stdout {
                   rlst::println_mpi!(mpi::topology::SimpleCommunicator::world(), "Id: {} - {}s", #ident, duration.as_secs_f64());
                }
             
                rlst::tracing::Tracing::add_duration(&#ident, duration);
                __result

            }



        }
        .into()
    } else {
        quote! {


        # (# attrs)*
        #vis #sig {
                #(#statements)*
        }


            }
        .into()
    }
}
