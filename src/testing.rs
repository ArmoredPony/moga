//! Fitness scores evaluation operators and utilities.

use executor::TestExecutor;
use rayon::prelude::*;

use crate::{
  execution::strategy::*,
  operator::{
    tag::TestOperatorTag,
    ParBatch,
    ParBatchOperator,
    ParEach,
    ParEachOperator,
  },
  score::Scores,
};

/// An operator that tests solution's fitness, evaluating an array of its
/// fitness scores.
///
/// The target value of a score, which it converges at, is considered to be 0.
/// Not `-infinity`, zero. `-5.0` is just as far from the ideal value as `5.0`.
/// If it does not align with your actual goal values, rewrite your objective
/// functions so they **do** converge at 0.
///
/// This crate's purpose is *multi-objective* optimizations, that's why tests
/// must return an *array* of values. If you want to return a single value,
/// wrap it in an array nonetheless.
///
/// Can be applied in parallel to each solution or to batches of solutions
/// by converting it into a parallelized operator with `par_each()` or
/// `par_batch()` methods.
///
/// # Examples
/// ```
/// # use moga::operator::*;
/// let t = |f: &f32| [f * 2.0]; // only one objective
/// let t = |f: &f32| [f + 1.0, f + 2.0, f + 3.0]; // 3 objectives
/// // or use an array of closures that return a single value
/// let t = [
///   |f: &f32| f + 1.0,
///   |f: &f32| f * f + 2.0,
///   |f: &f32| f * f * f + 3.0,
/// ];
/// t.par_batch();
/// ```
///
/// **Note that you always can implement this trait instead of using closures.**
pub trait Test<S, const N: usize> {
  /// Returns an array of fitness scores for given solution.
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

impl<S, const N: usize, T> ParEach<TestOperatorTag, S, N, 0> for T
where
  S: Sync,
  T: Test<S, N> + Sync,
{
}

impl<S, const N: usize, T> ParBatch<TestOperatorTag, S, N> for T
where
  S: Sync,
  T: Test<S, N> + Sync,
{
}

/// An operator that tests solutions' fitness, evaluating an array of fitness
/// scores for each solution.
///
/// The target value of a score, which it converges at, is considered to be 0.
/// Not `-infinity`, zero. `-5.0` is just as far from the ideal value as `5.0`.
/// If it does not align with your actual goal values, rewrite your objective
/// functions so they **do** converge at 0.
///
/// This crate's purpose is *multi-objective* optimizations, that's why tests
/// must return an *array* of values. If you want to return a single value,
/// wrap it in an array nonetheless.
///
/// # Examples
/// ```
/// # use moga::operator::*;
/// let t = |fs: &[f32]| fs.iter().map(|f| [f.log10(), f.sin()]).collect();
/// # let _: Vec<_> = t(&[]);
/// ```
///
/// **Note that you always can implement this trait instead of using closures.**
pub trait Tester<S, const N: usize> {
  /// Returns a vector of arrays of fitness scores for given solutions.
  /// The closer a score is to 0 - the better.
  ///
  /// # Panics
  ///
  /// Doesn't panic itself but will cause panic during optimization if this
  /// function returns a different number of scores than the number of solutions.
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

/// This private module prevents exposing the `Executor` to a user.
pub(crate) mod executor {
  use crate::score::Scores;

  /// An internal test executor.
  pub trait TestExecutor<S, const N: usize, ExecutionStrategy> {
    /// Executes tests optionally parallelizing operator's application.
    fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>>;
  }
}

impl<S, const N: usize, E> TestExecutor<S, N, CustomExecutionStrategy> for E
where
  E: Tester<S, N>,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    self.test(solutions)
  }
}

impl<const N: usize, S, T> TestExecutor<S, N, SequentialExecutionStrategy> for T
where
  T: Test<S, N>,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions.iter().map(|s| self.test(s)).collect()
  }
}

impl<const N: usize, S, T> TestExecutor<S, N, ParallelEachExecutionStrategy>
  for ParEachOperator<TestOperatorTag, S, T>
where
  S: Sync,
  T: Test<S, N> + Sync,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions
      .par_iter()
      .map(|s| self.operator().test(s))
      .collect()
  }
}

impl<const N: usize, S, T> TestExecutor<S, N, ParallelBatchExecutionStrategy>
  for ParBatchOperator<TestOperatorTag, S, T>
where
  S: Sync,
  T: Test<S, N> + Sync,
{
  fn execute_tests(&self, solutions: &[S]) -> Vec<Scores<N>> {
    let chunk_size = (solutions.len() / rayon::current_num_threads()).max(1);
    solutions
      .par_chunks(chunk_size)
      .flat_map_iter(|chunk| chunk.iter().map(|s| self.operator().test(s)))
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn takes_tester<ES, const N: usize, E: TestExecutor<Solution, N, ES>>(e: &E) {
    e.execute_tests(&[]);
  }

  #[test]
  fn test_test_from_closure() {
    let test = |v: &Solution| [v * 1.0, v * 2.0, v * 3.0];
    takes_tester(&test);
    takes_tester(&test.par_each());
    takes_tester(&test.par_batch());
  }

  #[test]
  fn test_test_from_closure_array() {
    let f1 = |v: &Solution| v * 1.0;
    let f2 = |v: &Solution| v * 2.0;
    let f3 = |v: &Solution| v * 3.0;
    let test = [f1, f2, f3];
    takes_tester(&test);
    takes_tester(&test.par_each());
    takes_tester(&test.par_batch());
  }

  #[test]
  fn test_tester_from_closure() {
    let tester = |solutions: &[Solution]| {
      solutions.iter().map(|_| [1.0, 2.0, 3.0]).collect()
    };
    takes_tester(&tester);
  }

  #[test]
  fn test_custom_test() {
    #[derive(Clone, Copy)]
    struct CustomTest {}
    impl<S> Test<S, 1> for CustomTest {
      fn test(&self, _: &S) -> Scores<1> {
        [0.0]
      }
    }

    let test = CustomTest {};
    takes_tester(&test);
    takes_tester(&test.par_each());
    takes_tester(&test.par_batch());
  }

  #[test]
  fn test_custom_tester() {
    struct CustomTester {}
    impl<S> Tester<S, 1> for CustomTester {
      fn test(&self, solutions: &[S]) -> Vec<Scores<1>> {
        solutions.iter().map(|_| [0.0]).collect()
      }
    }

    let tester = CustomTester {};
    takes_tester(&tester);
  }
}
