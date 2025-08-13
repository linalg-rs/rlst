use csv::ReaderBuilder;
use rlst::dense::linalg::qr::QrTolerance;
use rlst::{
    empty_array, rlst_dynamic_array2, DynamicArray, MultIntoResize, PrettyPrint, RawAccess, Shape,
};
use rlst::{Side, TransMode, TriangularMatrix, TriangularOperations, TriangularType};

fn load_csv_mat_into_rlst(path: &str) -> DynamicArray<f64, 2> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap();

    let mut rows_vec: Vec<Vec<f64>> = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        let row: Vec<f64> = record
            .iter()
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();
        rows_vec.push(row);
    }

    let m = rows_vec.len();
    let n = rows_vec[0].len();
    let mut mat = rlst_dynamic_array2!(f64, [m, n]);

    // Fill column-major (MATLAB order)
    for col in 0..n {
        for row in 0..m {
            mat.r_mut()[[row, col]] = rows_vec[row][col];
        }
    }

    mat
}
fn main() {
    let a = load_csv_mat_into_rlst("examples/A.csv");
    let pa_orig = load_csv_mat_into_rlst("examples/PA.csv");
    let r_orig = load_csv_mat_into_rlst("examples/R.csv");
    let q_orig = load_csv_mat_into_rlst("examples/Q.csv");

    // Print some stats to check
    println!("Generated matrix A of size {:?}", a.shape());

    let tol = 1e-4;
    let f = 1.01;
    let mut a_copy = empty_array();
    a_copy.r_mut().fill_from_resize(a.r());
    let rrqr = a_copy.r_mut().into_rrqr_alloc(
        rlst::RankRevealingQrType::SRRQR(f),
        rlst::RankParam::Tol(tol, QrTolerance::Abs),
    );

    println!("shapes: {:?}, {:?}", q_orig.shape(), rrqr.q.shape());
    let diff_r = (rrqr.r.r() - r_orig.r()).norm_fro();
    println!("diff r: {}", diff_r);
    let diff_q = (rrqr.q.r() - q_orig.r()).norm_fro();
    println!("diff q: {}", diff_q);

    rrqr.q.r().into_subview([15, 15], [5, 5]).pretty_print();
    q_orig.r().into_subview([15, 15], [5, 5]).pretty_print();

    let mut diff_mat = empty_array();
    diff_mat.r_mut().fill_from_resize(rrqr.q.r() - q_orig.r());

    let [rows, cols] = diff_mat.r().shape();

    let mut indices = Vec::new();
    let mut vals = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            if diff_mat.r()[[i, j]] > 1e-10 {
                indices.push((i, j));
                vals.push(diff_mat.r()[[i, j]]);
            }
        }
    }

    println!("indices: {:?}", indices);
    println!("vals: {:?}", vals);
    // Approximation error

    let [rows, cols] = a.r().shape();
    let dim = rows.min(cols);
    let rank = rrqr.rank;
    let mut a_app = empty_array();
    a_app
        .r_mut()
        .simple_mult_into_resize(rrqr.q.r(), rrqr.r.r());

    let mut perm_mat = rlst_dynamic_array2!(f64, [cols, cols]);
    perm_mat.set_zero();
    let mut view = perm_mat.r_mut();

    for (j, &col_idx) in rrqr.perm.iter().enumerate() {
        view[[col_idx, j]] = 1.0;
    }

    let mut pa = empty_array();
    pa.r_mut().simple_mult_into_resize(a.r(), perm_mat.r());

    let diff_pa = (pa.r() - pa_orig.r()).norm_fro();
    println!("diff pa: {}", diff_pa);

    let norm_a = pa.r().norm_fro();
    let err_abs = (pa.r() - a_app.r()).norm_fro();
    let error = err_abs / norm_a;

    let r11 = TriangularMatrix::<f64>::new(
        &rrqr.r.r().into_subview([0, 0], [rank, rank]),
        TriangularType::Upper,
    )
    .unwrap();

    let mut r12 = empty_array();
    r12.r_mut()
        .fill_from_resize(rrqr.r.r().into_subview([0, rank], [rank, dim - rank]));
    r11.solve(&mut r12, Side::Left, TransMode::NoTrans);

    let entry = r12.r().data().iter().map(|&x| x.abs()).fold(0.0, f64::max);

    println!("Approximation rank (fixed tol): {}\n", rank);
    println!("Relative approx. error: {}\n", error);
    println!("Maximum entry in inv(R11)*R12: {}\n", entry);
}
