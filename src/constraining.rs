//! Constraint violation scores evaluation operators and utilities.

use executor::ConstraintExecutor;
use rayon::prelude::*;

use crate::{
  execution::strategy::*,
  operator::{
    tag::ConstraintOperatorTag,
    ParBatch,
    ParBatchOperator,
    ParEach,
    ParEachOperator,
  },
  score::{Score, Scores},
};

/// An operator that constraints a solution, evaluating an array of its
/// violation scores.
///
/// The framework tries to minimize violation scores. If you want to maximize
/// them instead, then multiply by `-1`.
///
/// There are many ways to handle constraints in a genetic algorithm. The way
/// this framework handles them is by first comparing Pareto fronts of
/// violations, then fitness scores, when searching for least dominated solutions.
///
/// This approach *works* for multiple constraints just like it does for multiple
/// objectives, but you still can use a single value which is a (maybe weighted)
/// sum of your constraints. You are also free to ignore that operator whatsoever
/// and add a penalty score to fitness scores in your [`testing`] operators
/// instead, optimizers don't need constraints to be ran. This [paper] reviews
/// some ideas of constraint handling techniques.
///
/// Can be applied in parallel to each solution or to batches of solutions
/// by converting it into a parallelized operator with `par_each()` or
/// `par_batch()` methods.
///
/// # Examples
/// ```
/// # use moga::{operator::*, constraining::Constraint};
/// # fn takes_constraint<S, const N: usize, C: Constraint<S, N>>(c: C) {}
/// let c = |f: &f32| [-f.min(0.0)]; // requires the value to be greater than 0
/// // FIXME: fix examples
/// let c = |f: &f32| [0.0, 1.0]; // requires the value to be in interval [0, 1]
/// // or use an array of closures that return a single score
/// let c = [
///   |f: &f32| 0.0,
///   |f: &f32| 1.0,
/// ];
/// // FIXME: something needs to be done about that
/// // let c = par_batch();
/// # takes_constraint(c);
/// ```
///
/// **Note that you always can implement this trait instead of using closures.**
///
/// [`testing`]: crate::testing
/// [paper]: https://arxiv.org/pdf/2206.13802
pub trait Constraint<S, const N: usize> {
  /// Returns an array of violation scores for given solution.
  /// The lower the score - the better.
  fn constrain(&self, solution: &S) -> Scores<N>;
}

impl<S, const N: usize, F> Constraint<S, N> for [F; N]
where
  F: Fn(&S) -> Score,
{
  fn constrain(&self, solution: &S) -> Scores<N> {
    self.each_ref().map(|f| f(solution))
  }
}

impl<S, const N: usize, F> Constraint<S, N> for F
where
  F: Fn(&S) -> Scores<N>,
{
  fn constrain(&self, solution: &S) -> Scores<N> {
    self(solution)
  }
}

impl<S, const N: usize, C> ParEach<ConstraintOperatorTag, S, N, 0> for C
where
  S: Sync,
  C: Constraint<S, N> + Sync,
{
}

impl<S, const N: usize, C> ParBatch<ConstraintOperatorTag, S, N> for C
where
  S: Sync,
  C: Constraint<S, N> + Sync,
{
}

/// An operator that constraints a solution, evaluating an array of its
/// violation scores.
///
/// The framework tries to minimize violation scores. If you want to maximize
/// them instead, then multiply by `-1`.
///
/// There are many ways to handle constraints in a genetic algorithm. The way
/// this framework handles them is by first comparing Pareto fronts of
/// violations, then fitness scores, when searching for least dominated solutions.
///
/// This approach *works* for multiple constraints just like it does for multiple
/// objectives, but you still can use a single value which is a (maybe weighted)
/// sum of your constraints. You are also free to ignore that operator whatsoever
/// and add a penalty score to fitness scores in your [`testing`] operators
/// instead, optimizers don't need constraints to be ran. This [paper] reviews
/// some ideas of constraint handling techniques.
///
/// # Examples
/// ```
/// # use moga::operator::*;
/// // requires the value to be in interval [0, 1]
/// // FIXME: fix example
/// let c = |fs: &[f32]| fs.iter().map(|f| [0.0, 1.0]).collect();
/// # let _: Vec<_> = c(&[]);
/// ```
///
/// **Note that you always can implement this trait instead of using closures.**
///
/// [`testing`]: crate::testing
/// [paper]: https://arxiv.org/pdf/2206.13802
pub trait Constrainer<S, const N: usize> {
  /// Returns a vector of arrays of violation scores for given solutions.
  /// The framework favours solutions with lower scores.
  fn constrain(&self, solutions: &[S]) -> Vec<Scores<N>>;
}

impl<S, const N: usize, F> Constrainer<S, N> for F
where
  F: Fn(&[S]) -> Vec<Scores<N>>,
{
  fn constrain(&self, solutions: &[S]) -> Vec<Scores<N>> {
    self(solutions)
  }
}

/// This private module prevents exposing the `Executor` to a user.
pub(crate) mod executor {
  use crate::score::Scores;

  /// An internal constraint executor.
  pub trait ConstraintExecutor<S, const N: usize, ExecutionStrategy> {
    /// Executes constraints optionally parallelizing operator's application.
    fn execute_constraints(&self, solutions: &[S]) -> Vec<Scores<N>>;
  }
}

impl<S, const N: usize, E> ConstraintExecutor<S, N, CustomExecutionStrategy>
  for E
where
  E: Constrainer<S, N>,
{
  fn execute_constraints(&self, solutions: &[S]) -> Vec<Scores<N>> {
    self.constrain(solutions)
  }
}

impl<S, const N: usize, T> ConstraintExecutor<S, N, SequentialExecutionStrategy>
  for T
where
  T: Constraint<S, N>,
{
  fn execute_constraints(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions.iter().map(|s| self.constrain(s)).collect()
  }
}

impl<S, const N: usize, T>
  ConstraintExecutor<S, N, ParallelEachExecutionStrategy>
  for ParEachOperator<ConstraintOperatorTag, S, T>
where
  S: Sync,
  T: Constraint<S, N> + Sync,
{
  fn execute_constraints(&self, solutions: &[S]) -> Vec<Scores<N>> {
    solutions
      .par_iter()
      .map(|s| self.operator().constrain(s))
      .collect()
  }
}

impl<S, const N: usize, T>
  ConstraintExecutor<S, N, ParallelBatchExecutionStrategy>
  for ParBatchOperator<ConstraintOperatorTag, S, T>
where
  S: Sync,
  T: Constraint<S, N> + Sync,
{
  fn execute_constraints(&self, solutions: &[S]) -> Vec<Scores<N>> {
    let chunk_size = (solutions.len() / rayon::current_num_threads()).max(1);
    solutions
      .par_chunks(chunk_size)
      .flat_map_iter(|chunk| chunk.iter().map(|s| self.operator().constrain(s)))
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn takes_constrainer<
    ES,
    const N: usize,
    E: ConstraintExecutor<Solution, N, ES>,
  >(
    e: &E,
  ) {
    e.execute_constraints(&[]);
  }

  #[test]
  fn test_constraint_from_closure() {
    let constraint = |v: &Solution| [v * 2.0];
    takes_constrainer(&constraint);
    takes_constrainer(&constraint.par_each());
    takes_constrainer(&constraint.par_batch());
  }

  #[test]
  fn test_constrainer_from_closure() {
    let constrainer = |solutions: &[Solution]| {
      solutions.iter().map(|_| [1.0, 2.0, 3.0]).collect()
    };
    takes_constrainer(&constrainer);
  }

  #[test]
  fn test_custom_constraint() {
    #[derive(Clone, Copy)]
    struct CustomConstraint {}
    impl<S> Constraint<S, 1> for CustomConstraint {
      fn constrain(&self, _: &S) -> Scores<1> {
        [0.0]
      }
    }

    let constraint = CustomConstraint {};
    takes_constrainer(&constraint);
    takes_constrainer(&constraint.par_each());
    takes_constrainer(&constraint.par_batch());
  }

  #[test]
  fn test_custom_constrainer() {
    struct CustomConstrainer {}
    impl<S> Constrainer<S, 1> for CustomConstrainer {
      fn constrain(&self, solutions: &[S]) -> Vec<Scores<1>> {
        solutions.iter().map(|_| [0.0]).collect()
      }
    }

    let constrainer = CustomConstrainer {};
    takes_constrainer(&constrainer);
  }
}
