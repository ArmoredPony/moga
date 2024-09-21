//! Mutation operators and utilities.

use executor::MutationExecutor;
use rayon::prelude::*;

use crate::{
  execution::strategy::*,
  operator::{
    tag::MutationOperatorTag,
    ParBatch,
    ParBatchOperator,
    ParEach,
    ParEachOperator,
  },
};

/// An operator that mutates a single solution.
///
/// Can be applied in parallel to each solution or to batches of solutions
/// by converting it into a parallelized operator with `par_each()` or
/// `par_batch()` methods.
///
/// # Examples
/// ```
/// # use moga::operator::*;
/// let m = |f: &mut f32| *f *= 2.0;
/// let m = m.par_batch();
/// ```
///
/// **Note that you always can implement this trait instead of using closures.**
pub trait Mutation<S> {
  /// Mutates given solution.
  fn mutate(&self, solution: &mut S);
}

impl<S, F> Mutation<S> for F
where
  F: Fn(&mut S),
{
  fn mutate(&self, solution: &mut S) {
    self(solution)
  }
}

impl<S, M> ParEach<MutationOperatorTag, S, 0, 0> for M
where
  S: Sync + Send,
  M: Mutation<S> + Sync,
{
}

impl<S, M> ParBatch<MutationOperatorTag, S, 0> for M
where
  S: Sync + Send,
  M: Mutation<S> + Sync,
{
}

/// An operator that mutates all solutions.
///
/// # Examples
/// ```
/// let m = |fs: &mut [f32]| fs.iter_mut().for_each(|f| *f *= 2.0);
/// ```
///
/// **Note that you always can implement this trait instead of using closures.**
pub trait Mutator<S> {
  /// Mutates each solution in given solutions.
  fn mutate(&self, solutions: &mut [S]);
}

impl<S, F> Mutator<S> for F
where
  F: Fn(&mut [S]),
{
  fn mutate(&self, solutions: &mut [S]) {
    self(solutions)
  }
}

/// This private module prevents exposing the `Executor` to a user.
pub(crate) mod executor {
  /// An internal mutation executor.
  pub trait MutationExecutor<S, ExecutionStrategy> {
    /// Executes mutations optionally parallelizing operator's application.
    fn execute_mutations(&self, solutions: &mut [S]);
  }
}

impl<S, M> MutationExecutor<S, CustomExecutionStrategy> for M
where
  M: Mutator<S>,
{
  fn execute_mutations(&self, solutions: &mut [S]) {
    self.mutate(solutions)
  }
}

impl<S, M> MutationExecutor<S, SequentialExecutionStrategy> for M
where
  M: Mutation<S>,
{
  fn execute_mutations(&self, solutions: &mut [S]) {
    solutions.iter_mut().for_each(|s| self.mutate(s));
  }
}

impl<S, M> MutationExecutor<S, ParallelEachExecutionStrategy>
  for ParEachOperator<MutationOperatorTag, S, M>
where
  S: Sync + Send,
  M: Mutation<S> + Sync,
{
  fn execute_mutations(&self, solutions: &mut [S]) {
    solutions
      .par_iter_mut()
      .for_each(|s| self.operator().mutate(s));
  }
}

impl<S, M> MutationExecutor<S, ParallelBatchExecutionStrategy>
  for ParBatchOperator<MutationOperatorTag, S, M>
where
  S: Sync + Send,
  M: Mutation<S> + Sync,
{
  fn execute_mutations(&self, solutions: &mut [S]) {
    let chunk_size = (solutions.len() / rayon::current_num_threads()).max(1);
    solutions.par_chunks_mut(chunk_size).for_each(|chunk| {
      chunk.iter_mut().for_each(|s| self.operator().mutate(s))
    });
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn takes_mutator<ES, M: MutationExecutor<Solution, ES>>(m: &M) {
    m.execute_mutations(&mut []);
  }

  #[test]
  fn test_mutation_from_closure() {
    let mutation = |solution: &mut Solution| *solution *= 2.0;
    takes_mutator(&mutation);
    takes_mutator(&mutation.par_each());
    takes_mutator(&mutation.par_batch());
  }

  #[test]
  fn test_mutator_from_closure() {
    let mutator =
      |solutions: &mut [Solution]| solutions.iter_mut().for_each(|s| *s *= 2.0);
    takes_mutator(&mutator);
  }

  #[test]
  fn test_custom_mutation() {
    struct CustomMutation {}
    impl<S> Mutation<S> for CustomMutation {
      fn mutate(&self, _: &mut S) {}
    }

    let mutation = CustomMutation {};
    takes_mutator(&mutation);
  }

  #[test]
  fn test_custom_mutator() {
    struct CustomMutator {}
    impl<S> Mutator<S> for CustomMutator {
      fn mutate(&self, solutions: &mut [S]) {
        solutions.iter_mut().for_each(|_| {});
      }
    }
    let mutator = CustomMutator {};
    takes_mutator(&mutator);
  }
}
