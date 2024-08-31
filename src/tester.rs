use std::marker::PhantomData;

use rayon::prelude::*;

use crate::{execution::*, score::Scores};

/// Evaluates solution's fitness, calculating an array of scores.
/// `test` call returns an array of `Scores`. If you want to test a solution
/// against only one metric, wrap it in a single element array nonetheless.
///
/// This trait is implemented for closure of type `Fn(&S) -> [f32; N]`,
/// allowing for a simple and concise closure-to-test conversion.
///
/// # Examples
///
/// <pre> TODO: add them later </pre>
pub trait Test<S, const N: usize> {
  // TODO: add docs
  fn test(&self, solution: &S) -> Scores<N>;

  /// Creates a wrapper around `Test` that marks the given test to
  /// be executed in parallel for **each** solution.
  ///
  /// **Parallelization is implemented with [rayon]. As a result, for simple
  /// tests, parallelization may only decrease performance because of additional
  /// overhead introduced. Benchmark if in doubt.**
  fn par_each(self) -> ParEachTest<S, N, Self>
  where
    Self: Sized,
  {
    ParEachTest {
      test: self,
      _solution: PhantomData,
    }
  }

  /// Creates a wrapper around `Test` that marks the given test to
  /// be executed in parallel for each **batch** of solutions.
  ///
  /// **Parallelization is implemented with [rayon]. As a result, for simple
  /// tests, parallelization may only decrease performance because of additional
  /// overhead introduced. Benchmark if in doubt.**
  fn par_batch(self) -> ParBatchTest<S, N, Self>
  where
    Self: Sized,
  {
    ParBatchTest {
      test: self,
      _solution: PhantomData,
    }
  }
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

// TODO: add docs
pub struct ParEachTest<S, const N: usize, T: Test<S, N>> {
  test: T,
  _solution: PhantomData<S>,
}

// TODO: add docs
pub struct ParBatchTest<S, const N: usize, T: Test<S, N>> {
  test: T,
  _solution: PhantomData<S>,
}

/// Tests performance of each solution, calculating their scores.
/// The target score of each objective in test is deemed to be 0.
pub trait Tester<S, const N: usize> {
  /// Returns a vector of performance scores for each solution.
  /// The closer a score to 0 - the better.
  fn test(&self, solutions: &[S]) -> Vec<Scores<N>>;
}

// TODO: add docs
pub trait TestExecutor<S, const N: usize, ExecutionStrategy> {
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>>;
}

impl<S, const N: usize, F> Tester<S, N> for F
where
  F: Fn(&[S]) -> Vec<Scores<N>>,
{
  fn test(&self, solutions: &[S]) -> Vec<Scores<N>> {
    self(solutions)
  }
}

impl<S, const N: usize, E> TestExecutor<S, N, CustomExecution> for E
where
  E: Tester<S, N>,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    Tester::test(self, solutions)
  }
}

impl<const N: usize, S, O> TestExecutor<S, N, SequentialExecution> for O
where
  O: Test<S, N>,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions.iter().map(|s| Test::test(self, s)).collect()
  }
}

impl<const N: usize, S, O> TestExecutor<S, N, ParallelEachExecution>
  for ParEachTest<S, N, O>
where
  S: Sync,
  O: Test<S, N> + Sync,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions.par_iter().map(|s| self.test.test(s)).collect()
  }
}

impl<const N: usize, S, O> TestExecutor<S, N, ParallelBatchExecution>
  for ParBatchTest<S, N, O>
where
  S: Sync,
  O: Test<S, N> + Sync,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    let chunk_size = (solutions.len() / rayon::current_num_threads()).max(1);
    solutions
      .par_chunks(chunk_size)
      .flat_map_iter(|chunk| chunk.iter().map(|s| self.test.test(s)))
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
