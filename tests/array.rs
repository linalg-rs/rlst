//! Basic array functionality.

use rlst::{self, ContainerTypeHint, EvaluateArray};

#[test]
fn test_array_eval() {
    let mut arr: rlst::Array<_, 2> = rlst::rlst_static_array!(f64, 4, 5);

    arr.fill_from_seed_normally_distributed(0);
    let arr = arr.eval();
    let type_hint = arr.type_hint_as_str();
    assert_eq!(type_hint, "Stack");

    assert_eq!(
        arr.with_container_type::<rlst::Heap>().type_hint_as_str(),
        "Heap"
    );

    let mut arr = rlst::rlst_dynamic_array!(f64, [4, 5]);

    arr.fill_from_seed_normally_distributed(0);
    let arr = arr.eval();
    let type_hint = arr.type_hint_as_str();
    assert_eq!(type_hint, "Heap");

    let type_hint = arr
        .with_container_type::<rlst::Stack<20>>()
        .type_hint_as_str();

    assert_eq!(type_hint, "Stack");
}
