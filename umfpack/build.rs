use bindgen;
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(std::env::var("DEP_SUITESPARSE_ROOT").unwrap());

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/wrapper.h")
        // Need to block some types in math.h that cause problems with bindgen
        // on Linux. See https://github.com/rust-lang/rust-bindgen/issues/984
        .blocklist_type("FP_NAN")
        .blocklist_type("FP_INFINITE")
        .blocklist_type("FP_ZERO")
        .blocklist_type("FP_SUBNORMAL")
        .blocklist_type("FP_NORMAL")
        // Add an include path
        .clang_arg(format!("-I{}", root.join("include").display()))
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
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
