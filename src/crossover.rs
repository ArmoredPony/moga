use rayon::prelude::*;

/// Creates new solutions from previously selected.
/// This operator spawns `M` offsprings from `N` parents.
pub trait Crossover<const N: usize, const M: usize, S> {
  /// Takes a slice of selected solutions and returns created offsprings.
  fn create(&self, solutions: &[S]) -> Vec<S>;
}

impl<S, F> Crossover<1, 1, S> for F
where
  S: Send + Sync,
  F: Fn(&S) -> S + Sync,
{
  fn create(&self, solutions: &[S]) -> Vec<S> {
    solutions.par_iter().map(self).collect()
  }
}

impl<S, F> Crossover<2, 1, S> for F
where
  S: Send + Sync,
  F: Fn(&S, &S) -> S + Sync,
{
  fn create(&self, solutions: &[S]) -> Vec<S> {
    if solutions.is_empty() {
      return vec![];
    }
    (0..solutions.len() - 1)
      .flat_map(|i| {
        (i + 1..solutions.len()).map(move |j| (&solutions[i], &solutions[j]))
      })
      .par_bridge()
      .map(|(a, b)| self(a, b))
      .collect()
  }
}

// tuple-to-array conversion can be implemented with a macro
impl<S, F> Crossover<2, 2, S> for F
where
  S: Send + Sync,
  F: Fn(&S, &S) -> (S, S) + Sync,
{
  fn create(&self, solutions: &[S]) -> Vec<S> {
    if solutions.is_empty() {
      return vec![];
    }
    (0..solutions.len() - 1)
      .flat_map(|i| {
        (i + 1..solutions.len()).map(move |j| (&solutions[i], &solutions[j]))
      })
      .par_bridge()
      .flat_map(|(a, b)| <[S; 2]>::from(self(a, b)))
      .collect()
  }
}

impl<S, F> Crossover<{ usize::MAX }, { usize::MAX }, S> for F
where
  F: Fn(&[S]) -> Vec<S>,
{
  fn create(&self, solutions: &[S]) -> Vec<S> {
    self(solutions)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f64;

  const fn as_crossover<
    const N: usize,
    const M: usize,
    C: Crossover<N, M, Solution>,
  >(
    _: &C,
  ) {
  }

  #[test]
  fn test_crossover_1_to_1() {
    let c = |s: &Solution| s.to_owned() + 1.0;
    as_crossover(&c);
    let parents: Vec<_> = (0..100).map(Solution::from).collect();
    let offsprings = c.create(&parents);
    assert_eq!(parents.len(), offsprings.len());
    assert_eq!(c.create(&[]), &[]);
  }

  #[test]
  fn test_crossover_2_to_1() {
    let c = |a: &Solution, b: &Solution| a + b;
    as_crossover(&c);
    let parents: Vec<_> = (0..100).map(Solution::from).collect();
    let offsprings = c.create(&parents);
    assert_eq!(offsprings.len(), (0..parents.len()).sum());
    assert_eq!(c.create(&[]), &[]);
    assert_eq!(c.create(&[1.0]), &[]);
    assert_eq!(c.create(&[1.0, 2.0]), &[3.0]);
  }

  #[test]
  fn test_crossover_2_to_2() {
    let c = |a: &Solution, b: &Solution| (a + b, a - b);
    as_crossover(&c);
    let parents: Vec<_> = (0..100).map(Solution::from).collect();
    let offsprings = c.create(&parents);
    assert_eq!(offsprings.len(), (0..parents.len()).sum::<usize>() * 2);
    assert_eq!(c.create(&[]), &[]);
    assert_eq!(c.create(&[1.0]), &[]);
    assert_eq!(c.create(&[1.0, 2.0]), &[3.0, -1.0]);
  }
}
