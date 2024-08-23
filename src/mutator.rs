/// Mutates solution.
pub trait Mutator<S> {
  /// Takes a solution and mutates it.
  fn mutate(&self, solution: &mut S);
}
