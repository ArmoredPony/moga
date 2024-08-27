pub(crate) mod pareto;

// An alias for objective score. The target value of a score is 0.
pub(crate) type Score = f32;

/// An alias for array of `N` `Score` values.
pub(crate) type Scores<const N: usize> = [Score; N];

/// Represents solution's performance scores. `N` is a number of objectives.
/// The target score of each objective is deemed to be 0.
pub trait Objectives<S, const N: usize> {
  /// Returns solution's performance scores. The closer a score to 0 - the
  /// better.
  fn evaluate(&self, solution: &S) -> Scores<N>;
}

impl<const N: usize, S, F> Objectives<S, N> for [F; N]
where
  F: Fn(&S) -> f32,
{
  fn evaluate(&self, solution: &S) -> Scores<N> {
    self.each_ref().map(|f| f(solution))
  }
}

impl<const N: usize, S, F> Objectives<S, N> for F
where
  F: Fn(&S) -> Scores<N>,
{
  fn evaluate(&self, solution: &S) -> Scores<N> {
    self(solution)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn as_objective<const N: usize, O: Objectives<Solution, N>>(_: &O) {}

  #[test]
  fn test_objective_from_closure() {
    let o = |v: &Solution| [v * 1.0, v * 2.0, v * 3.0];
    as_objective(&o);
    assert_eq!(o.evaluate(&1.0), [1.0, 2.0, 3.0]);
  }

  #[test]
  fn test_objective_from_closure_array() {
    let o1 = |v: &Solution| v * 1.0;
    let o2 = |v: &Solution| v * 2.0;
    let o3 = |v: &Solution| v * 3.0;
    let os = [o1, o2, o3];
    as_objective(&os);
    assert_eq!(os.evaluate(&1.0), [1.0, 2.0, 3.0])
  }
}
