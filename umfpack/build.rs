use std::path::PathBuf;

// NOTE: If FP_NORMAL and other symbols from math.h cause problems use
// the solution from https://github.com/rust-lang/rust-bindgen/issues/984

fn main() {
    let root = PathBuf::from(std::env::var("DEP_SUITESPARSE_ROOT").unwrap());

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/wrapper.h")
        // Add an include path
        .clang_arg(format!("-I{}", root.join("include").display()))
        .allowlist_function("umfpack.*")
        .allowlist_type("UMFPACK.*")
        .allowlist_var("UMFPACK.*")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    //println!("Out path: {}", out_path.to_str().unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
