use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rlst::EvaluateObject;

fn main() {
    // Declare an array either as
    let mut arr1 = rlst::DynArray::<f64, 2>::from_shape([3, 5]);
    // or as macro via
    let mut arr2 = rlst::rlst_dynamic_array!(f64, [3, 5]);
    // The second variant only works for explicit arrays. We cannot say
    // let shape = [3, 5];
    // let mut arr3 = rlst::rlst_dynamic_array!(f64, shape);
    // since currently the procedural macro always expects an array-like expression.
    // This is because it needs to know the number of dimensions at compile time.

    // Let us now fill the arrays.

    let mut rng = ChaCha8Rng::seed_from_u64(0);

    // The following uses equally distributed random numbers from 0 to 1.
    arr1.fill_from_equally_distributed(&mut rng);
    // The following uses normally distributed random numbers with mean 0 and variance 1.
    arr2.fill_from_standard_normal(&mut rng);

    // We can manually access array elements.
    arr1[[0, 1]] = 5.0;

    // Let's compute an array expression.

    let res = arr1.r() + 5.0 * arr2.r().exp();

    // To evaluate into a new array we use the `EvaluateObject` trait.

    let _arr3 = res.eval();
}
