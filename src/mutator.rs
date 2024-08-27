/// Mutates solution.
pub trait Mutator<S> {
  /// Takes a solution and mutates it.
  fn mutate(&self, solution: &mut S);
}

/// `Mutator` that doesn't mutate given value.
pub struct NoMutator();

impl<S> Mutator<S> for NoMutator {
  fn mutate(&self, _: &mut S) {}
}

impl<S, F> Mutator<S> for F
where
  F: Fn(&mut S),
{
  fn mutate(&self, solution: &mut S) {
    self(solution)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn as_mutator<M: Mutator<Solution>>(_: &M) {}

  #[test]
  fn test_no_mutator() {
    let m = NoMutator();
    as_mutator(&m);
  }

  #[test]
  fn test_mutator_from_fn() {
    let m = |solution: &mut Solution| *solution *= 2.0;
    as_mutator(&m);
  }
}
