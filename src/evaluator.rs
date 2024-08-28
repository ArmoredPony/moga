pub(crate) mod pareto;

// An alias for objective score. The target value of a score is 0.
pub(crate) type Score = f32;

/// An alias for array of `N` `Score` values.
pub(crate) type Scores<const N: usize> = [Score; N];

/// Evaluates solution's performance scores. `N` is a number of objectives.
/// The target score of each objective is deemed to be 0.
pub trait Evaluator<S, const N: usize> {
  /// Returns solution's performance scores. The closer a score to 0 - the
  /// better.
  fn evaluate(&self, solution: &S) -> Scores<N>;
}

impl<const N: usize, S, F> Evaluator<S, N> for [F; N]
where
  F: Fn(&S) -> f32,
{
  fn evaluate(&self, solution: &S) -> Scores<N> {
    self.each_ref().map(|f| f(solution))
  }
}

impl<const N: usize, S, F> Evaluator<S, N> for F
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

  fn as_evaluator<const N: usize, O: Evaluator<Solution, N>>(_: &O) {}

  #[test]
  fn test_evaluator_from_closure() {
    let e = |v: &Solution| [v * 1.0, v * 2.0, v * 3.0];
    as_evaluator(&e);
    assert_eq!(e.evaluate(&1.0), [1.0, 2.0, 3.0]);
  }

  #[test]
  fn test_evaluator_from_closure_array() {
    let e1 = |v: &Solution| v * 1.0;
    let e2 = |v: &Solution| v * 2.0;
    let e3 = |v: &Solution| v * 3.0;
    let es = [e1, e2, e3];
    as_evaluator(&es);
    assert_eq!(es.evaluate(&1.0), [1.0, 2.0, 3.0])
  }
}
