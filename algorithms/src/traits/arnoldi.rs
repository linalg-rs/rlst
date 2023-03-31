/// Arnoldi implemented for operators

pub trait Arnoldi {
    type OrthogonalBasis;
    type HessenbergMatrix;
    type Domain;
    type Range: Inner<T = Self::T>;
    type T: Scalar;
    type Operator: Apply<Self::Domain, T = Self::T, Range = Self::Range>;

    fn initialize(start: Self::Domain, max_steps: IndexType) -> Self;
    fn arnoldi_step(&self, step_count: IndexType);

    fn orthogonal_basis(&self) -> &Self::OrthogonalBasis;

    fn hessenberg_matrix(&self) -> &Self::HessenbergMatrix;
}
