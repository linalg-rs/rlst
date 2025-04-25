use rlst::rlst_dynamic_array2;

#[test]
fn inverse_100() {
    let size = 100;

    let mut mat = rlst_dynamic_array2!(f64, [size, size]);

    for index in 0..size {
        mat[[index, index]] = 1.0;
    }

    mat.r_mut()
        .into_inverse_alloc()
        .expect("Inverse could not be computed.");
}
