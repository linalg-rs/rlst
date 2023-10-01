//! Relation between numbers represented as types and traits

pub struct NumberType<const N: usize>;

pub trait IsGreaterByOne<const N: usize> {}
pub trait IsGreaterZero {}

impl IsGreaterByOne<0> for NumberType<1> {}
impl IsGreaterByOne<1> for NumberType<2> {}
impl IsGreaterByOne<2> for NumberType<3> {}
impl IsGreaterByOne<3> for NumberType<4> {}
impl IsGreaterByOne<4> for NumberType<5> {}
impl IsGreaterByOne<5> for NumberType<6> {}
impl IsGreaterByOne<6> for NumberType<7> {}
impl IsGreaterByOne<7> for NumberType<8> {}
impl IsGreaterByOne<8> for NumberType<9> {}
impl IsGreaterByOne<9> for NumberType<10> {}
impl IsGreaterByOne<10> for NumberType<11> {}
impl IsGreaterByOne<11> for NumberType<12> {}
impl IsGreaterByOne<12> for NumberType<13> {}
impl IsGreaterByOne<13> for NumberType<14> {}
impl IsGreaterByOne<14> for NumberType<15> {}
impl IsGreaterByOne<15> for NumberType<16> {}
impl IsGreaterByOne<16> for NumberType<17> {}
impl IsGreaterByOne<17> for NumberType<18> {}
impl IsGreaterByOne<18> for NumberType<19> {}
impl IsGreaterByOne<19> for NumberType<20> {}

impl IsGreaterZero for NumberType<1> {}
impl IsGreaterZero for NumberType<2> {}
impl IsGreaterZero for NumberType<3> {}
impl IsGreaterZero for NumberType<4> {}
impl IsGreaterZero for NumberType<5> {}
impl IsGreaterZero for NumberType<6> {}
impl IsGreaterZero for NumberType<7> {}
impl IsGreaterZero for NumberType<8> {}
impl IsGreaterZero for NumberType<9> {}
impl IsGreaterZero for NumberType<10> {}
impl IsGreaterZero for NumberType<11> {}
impl IsGreaterZero for NumberType<12> {}
impl IsGreaterZero for NumberType<13> {}
impl IsGreaterZero for NumberType<14> {}
impl IsGreaterZero for NumberType<15> {}
impl IsGreaterZero for NumberType<16> {}
impl IsGreaterZero for NumberType<17> {}
impl IsGreaterZero for NumberType<18> {}
impl IsGreaterZero for NumberType<19> {}
impl IsGreaterZero for NumberType<120> {}
