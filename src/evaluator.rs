use std::marker::PhantomData;

use rayon::{prelude::*, ThreadPool};

pub(crate) mod pareto;

/// Sequential execution strategy, i.e. no parallelization involved.
pub enum Sequential {}
/// Parallel execution strategy, parallelizes objective testing for **each**
/// solution.
pub enum ParallelEach {}
/// Parallel execution strategy, parallelizes objective testing for a **batch**
/// of solutions. By default, the crate tries to split the work equally for
/// each available thread.
pub enum ParallelBatch {}

// An alias for objective score. The target value of a score is 0.
pub type Score = f32;

/// An alias for array of `N` `Score` values.
pub type Scores<const N: usize> = [Score; N];

/// Tests a solution, calculating an array of scores.
/// Each `test` returns an array of `Scores`. If you want to test a solution
/// against only one metric, wrap it in a single element array nonetheless.
///
/// This trait is implemented for closure of type `Fn(&S) -> [f32; N]`,
/// allowing for a simple and consise closure-to-objective conversion.
///
/// # Examples
///
/// <pre> add them later </pre>
pub trait Objective<S, const N: usize> {
  fn test(&self, solution: &S) -> Scores<N>;
}

impl<S, const N: usize, F> Objective<S, N> for [F; N]
where
  F: Fn(&S) -> f32,
{
  fn test(&self, solution: &S) -> Scores<N> {
    self.each_ref().map(|f| f(solution))
  }
}

impl<S, const N: usize, F> Objective<S, N> for F
where
  F: Fn(&S) -> Scores<N>,
{
  fn test(&self, solution: &S) -> Scores<N> {
    self(solution)
  }
}

/// A wrapper around `Objective` that marks the given objective to be executed
/// in parallel for **each** solution.
///
/// **Parallelization is implemented with [rayon]. As a result, for simple
/// tests, parallelization may only decrease performance because of additional
/// overhead introduced. Benchmark if in doubt.**
pub struct ParEachObjective<S, const N: usize, O: Objective<S, N>> {
  objective: O,
  _solution: PhantomData<S>,
}

/// A wrapper around `Objective` that marks the given objective to be executed
/// in parallel for each **batch** of solutions.
///
/// **Parallelization is implemented with [rayon]. As a result, for simple
/// tests, parallelization may only decrease performance because of additional
/// overhead introduced. Benchmark if in doubt.**
pub struct ParBatchObjective<S, const N: usize, O: Objective<S, N>> {
  objective: O,
  _solution: PhantomData<S>,
}

/// Implements the conversion of an `Objective` to a parallelized objective.
/// Selected method allows you to specify the kind of parallelization.
pub trait IntoParObjective<S, const N: usize>
where
  Self: Objective<S, N> + Sized,
{
  fn par_each(self) -> ParEachObjective<S, N, Self> {
    ParEachObjective {
      objective: self,
      _solution: PhantomData,
    }
  }

  fn par_batch(self) -> ParBatchObjective<S, N, Self> {
    ParBatchObjective {
      objective: self,
      _solution: PhantomData,
    }
  }
}

impl<S, const N: usize, O: Objective<S, N>> IntoParObjective<S, N> for O {}

/// Evaluates performance scores for each solution. `N` is a number of
/// objectives. The target score of each objective is deemed to be 0.
///
/// Default execution strategy does not involve parallelization. To enable it
/// for an `Objective`, convert it into `Par...Objective` with `par_...`
/// methods. See [`IntoParObjective`] for available conversions.
pub trait Evaluator<ExecutionStrategy, S, const N: usize> {
  /// Returns a vector performance scores for each solution.
  /// The closer a score to 0 - the better.
  fn evaluate(&self, solutions: &[S]) -> Vec<Scores<N>>;
}

impl<const N: usize, S, O> Evaluator<Sequential, S, N> for O
where
  O: Objective<S, N>,
{
  fn evaluate(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions.iter().map(|s| self.test(s)).collect()
  }
}

impl<const N: usize, S, O> Evaluator<ParallelEach, S, N>
  for ParEachObjective<S, N, O>
where
  S: Sync,
  O: Objective<S, N> + Sync,
{
  fn evaluate(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions
      .par_iter()
      .map(|s| self.objective.test(s))
      .collect()
  }
}

impl<const N: usize, S, O> Evaluator<ParallelBatch, S, N>
  for ParBatchObjective<S, N, O>
where
  S: Sync,
  O: Objective<S, N> + Sync,
{
  fn evaluate(&self, solutions: &[S]) -> Vec<Scores<N>> {
    let chunk_size = solutions.len() / rayon::current_num_threads();
    solutions
      .par_chunks(chunk_size)
      .flat_map_iter(|chunk| chunk.iter().map(|s| self.objective.test(s)))
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn as_evaluator<ES, const N: usize, E: Evaluator<ES, Solution, N>>(e: &E) {
    e.evaluate(&[1.0, 2.0, 3.0]);
  }

  #[test]
  fn test_evaluator_from_closure() {
    let e = |v: &Solution| [v * 1.0, v * 2.0, v * 3.0];
    // as_evaluator(&e);
    // as_evaluator(&e.par_each());

    for _ in 0..10e4 as usize {
      as_evaluator(&e.par_each());
    }
    // assert_eq!(e.evaluate(&1.0), [1.0, 2.0, 3.0]);
  }

  // #[test]
  // fn test_evaluator_from_closure_array() {
  //   let e1 = |v: &Solution| v * 1.0;
  //   let e2 = |v: &Solution| v * 2.0;
  //   let e3 = |v: &Solution| v * 3.0;
  //   let es = [e1, e2, e3];
  //   as_evaluator(&es);
  //   assert_eq!(es.evaluate(&1.0), [1.0, 2.0, 3.0])
  // }
}
