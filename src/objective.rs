/// Calculates solution's performance scores. `N` is a number of performance
/// metrics.
pub trait Objective<const N: usize, S> {
  /// Returns solution's performance scores. The closer to 0 - the better.
  fn evaluate(&self, solution: &S) -> [f32; N];
}

impl<const N: usize, S, F> Objective<N, S> for [F; N]
where
  F: Fn(&S) -> f32,
{
  fn evaluate(&self, solution: &S) -> [f32; N] {
    self.each_ref().map(|f| f(solution))
  }
}

impl<S, F> Objective<1, S> for F
where
  F: Fn(&S) -> f32,
{
  fn evaluate(&self, solution: &S) -> [f32; 1] {
    [self(solution)]
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn as_objective<const N: usize, O: Objective<N, Solution>>(_: O) {}

  #[test]
  fn test_objective_from_closure() {
    let o = |v: &Solution| v * 2.0;
    as_objective(o);
    assert_eq!(o.evaluate(&1.0), [2.0]);
  }

  #[test]
  fn test_objective_from_closures() {
    let o1 = |v: &Solution| v * 1.0;
    let o2 = |v: &Solution| v * 2.0;
    let o3 = |v: &Solution| v * 3.0;
    let os = [o1, o2, o3];
    as_objective(os);
    assert_eq!(os.evaluate(&1.0), [1.0, 2.0, 3.0])
  }
}
