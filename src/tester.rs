use rayon::prelude::*;

use crate::{execution::*, score::Scores};

/// Evaluates solution's fitness, calculating an array of scores.
/// `test` call returns an array of `Scores`. If you want to test a solution
/// against only one metric, wrap it in a single element array nonetheless.
/// The target score of each objective in test is deemed to be 0.
///
/// This trait is implemented for closure of type `Fn(&S) -> [f32; N]`,
/// allowing for a simple and concise closure-to-test conversion.
///
/// # Examples
///
/// <pre> TODO: add them later </pre>
pub trait Test<S, const N: usize> {
  /// Returns performance scores for given solution.
  /// The closer a score is to 0 - the better.
  fn test(&self, solution: &S) -> Scores<N>;
}

impl<S, const N: usize, F> Test<S, N> for [F; N]
where
  F: Fn(&S) -> f32,
{
  fn test(&self, solution: &S) -> Scores<N> {
    self.each_ref().map(|f| f(solution))
  }
}

impl<S, const N: usize, F> Test<S, N> for F
where
  F: Fn(&S) -> Scores<N>,
{
  fn test(&self, solution: &S) -> Scores<N> {
    self(solution)
  }
}

impl<S, const N: usize, T> IntoPar<S, N> for T where T: Test<S, N> {}

/// Tests performance of each solution, calculating their scores.
/// The target score of each objective in test is deemed to be 0.
pub trait Tester<S, const N: usize> {
  /// Returns a vector of performance scores for each solution.
  /// The closer a score is to 0 - the better.
  fn test(&self, solutions: &[S]) -> Vec<Scores<N>>;
}

impl<S, const N: usize, F> Tester<S, N> for F
where
  F: Fn(&[S]) -> Vec<Scores<N>>,
{
  fn test(&self, solutions: &[S]) -> Vec<Scores<N>> {
    self(solutions)
  }
}

// TODO: add docs
// TODO: make private
pub trait TestExecutor<S, const N: usize, ExecutionStrategy> {
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>>;
}

impl<S, const N: usize, E> TestExecutor<S, N, CustomExecution> for E
where
  E: Tester<S, N>,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    Tester::test(self, solutions)
  }
}

impl<const N: usize, S, T> TestExecutor<S, N, SequentialExecution> for T
where
  T: Test<S, N>,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions.iter().map(|s| Test::test(self, s)).collect()
  }
}

impl<const N: usize, S, T> TestExecutor<S, N, ParallelEachExecution>
  for ParEach<S, T>
where
  S: Sync,
  T: Test<S, N> + Sync,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions.par_iter().map(|s| self.test(s)).collect()
  }
}

impl<const N: usize, S, T> TestExecutor<S, N, ParallelBatchExecution>
  for ParBatch<S, T>
where
  S: Sync,
  T: Test<S, N> + Sync,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    let chunk_size = (solutions.len() / rayon::current_num_threads()).max(1);
    solutions
      .par_chunks(chunk_size)
      .flat_map_iter(|chunk| chunk.iter().map(|s| self.test(s)))
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn as_tester<ES, const N: usize, E: TestExecutor<Solution, N, ES>>(e: &E) {
    e.execute_tests(&[]);
  }

  #[test]
  fn test_test_from_closure() {
    let test = |v: &Solution| [v * 1.0, v * 2.0, v * 3.0];
    as_tester(&test);
    as_tester(&test.par_each());
    as_tester(&test.par_batch());
  }

  #[test]
  fn test_test_from_closure_array() {
    let f1 = |v: &Solution| v * 1.0;
    let f2 = |v: &Solution| v * 2.0;
    let f3 = |v: &Solution| v * 3.0;
    let test = [f1, f2, f3];
    as_tester(&test);
    as_tester(&test.par_each());
    as_tester(&test.par_batch());
  }

  #[test]
  fn test_tester_from_closure() {
    let tester = |solutions: &[Solution]| {
      solutions.iter().map(|_| [1.0, 2.0, 3.0]).collect()
    };
    as_tester(&tester);
  }

  struct CustomTester {} // always returns [0.0] for each solution
  impl<S> Tester<S, 1> for CustomTester {
    fn test(&self, solutions: &[S]) -> Vec<Scores<1>> {
      solutions.iter().map(|_| [0.0]).collect()
    }
  }

  #[test]
  fn test_custom_tester() {
    let tester = CustomTester {};
    as_tester(&tester);
  }
}
