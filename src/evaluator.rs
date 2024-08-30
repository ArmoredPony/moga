use std::marker::PhantomData;

use rayon::prelude::*;

use crate::execution::*;

pub(crate) mod pareto;

// An alias for objective score. The target value of a score is 0.
pub type Score = f32;

/// An alias for array of `N` `Score` values.
pub type Scores<const N: usize> = [Score; N];

/// Evaluates solution's fitness, calculating an array of scores.
/// `evaluate` returns an array of `Scores`. If you want to test a solution
/// against only one metric, wrap it in a single element array nonetheless.
///
/// This trait is implemented for closure of type `Fn(&S) -> [f32; N]`,
/// allowing for a simple and consise closure-to-objective conversion.
///
/// # Examples
///
/// <pre> add them later </pre>
pub trait Objective<S, const N: usize> {
  // TODO: add docs
  fn evaluate(&self, solution: &S) -> Scores<N>;

  // TODO: add docs
  fn par_each(self) -> ParEachObjective<S, N, Self>
  where
    Self: Sized,
  {
    ParEachObjective {
      objective: self,
      _solution: PhantomData,
    }
  }

  // TODO: add docs
  fn par_batch(self) -> ParBatchObjective<S, N, Self>
  where
    Self: Sized,
  {
    ParBatchObjective {
      objective: self,
      _solution: PhantomData,
    }
  }
}

impl<S, const N: usize, F> Objective<S, N> for [F; N]
where
  F: Fn(&S) -> f32,
{
  fn evaluate(&self, solution: &S) -> Scores<N> {
    self.each_ref().map(|f| f(solution))
  }
}

impl<S, const N: usize, F> Objective<S, N> for F
where
  F: Fn(&S) -> Scores<N>,
{
  fn evaluate(&self, solution: &S) -> Scores<N> {
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

/// Evaluates performance scores for each solution. `N` is a number of
/// objectives. The target score of each objective is deemed to be 0.
pub trait Evaluator<S, const N: usize> {
  /// Returns a vector performance scores for each solution.
  /// The closer a score to 0 - the better.
  fn evaluate(&self, solutions: &[S]) -> Vec<Scores<N>>;
}

pub trait EvaluatorExecutor<ExecutionStrategy, S, const N: usize> {
  fn evaluate(&self, solutions: &[S]) -> Vec<Scores<N>>;
}

impl<S, const N: usize, E> EvaluatorExecutor<CustomExecution, S, N> for E
where
  E: Evaluator<S, N>,
{
  fn evaluate(&self, solutions: &[S]) -> Vec<Scores<N>> {
    Evaluator::evaluate(self, solutions)
  }
}

impl<const N: usize, S, O> EvaluatorExecutor<SequentialExecution, S, N> for O
where
  O: Objective<S, N>,
{
  fn evaluate(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions.iter().map(|s| self.evaluate(s)).collect()
  }
}

impl<const N: usize, S, O> EvaluatorExecutor<ParallelEachExecution, S, N>
  for ParEachObjective<S, N, O>
where
  S: Sync,
  O: Objective<S, N> + Sync,
{
  fn evaluate(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions
      .par_iter()
      .map(|s| self.objective.evaluate(s))
      .collect()
  }
}

impl<const N: usize, S, O> EvaluatorExecutor<ParallelBatchExecution, S, N>
  for ParBatchObjective<S, N, O>
where
  S: Sync,
  O: Objective<S, N> + Sync,
{
  fn evaluate(&self, solutions: &[S]) -> Vec<Scores<N>> {
    let chunk_size = (solutions.len() / rayon::current_num_threads()).max(1);
    solutions
      .par_chunks(chunk_size)
      .flat_map_iter(|chunk| chunk.iter().map(|s| self.objective.evaluate(s)))
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn as_evaluator<ES, const N: usize, E: EvaluatorExecutor<ES, Solution, N>>(
    e: &E,
  ) {
    e.evaluate(&[]);
  }

  #[test]
  fn test_evaluator_from_closure() {
    let e = |v: &Solution| [v * 1.0, v * 2.0, v * 3.0];
    as_evaluator(&e);
    as_evaluator(&e.par_each());
    as_evaluator(&e.par_batch());
  }

  #[test]
  fn test_evaluator_from_closure_array() {
    let o1 = |v: &Solution| v * 1.0;
    let o2 = |v: &Solution| v * 2.0;
    let o3 = |v: &Solution| v * 3.0;
    let e = [o1, o2, o3];
    as_evaluator(&e);
    as_evaluator(&e.par_each());
    as_evaluator(&e.par_batch());
  }

  // TODO: test custom `Evaluator`
}
