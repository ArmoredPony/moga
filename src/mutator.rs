use rayon::prelude::*;

use crate::{
  execution::*,
  operator::{IntoParOperator, MutationOperatorTag, ParBatch, ParEach},
};

/// Mutates a solution.
pub trait MutationOperator<S> {
  /// Takes a solution and mutates it.
  fn mutate(&self, solution: &mut S);
}

impl<S, F> MutationOperator<S> for F
where
  F: Fn(&mut S),
{
  fn mutate(&self, solution: &mut S) {
    self(solution)
  }
}

impl<S, M> IntoParOperator<MutationOperatorTag, S, 0> for M where
  M: MutationOperator<S>
{
}

/// Mutates solutions.
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

// TODO: add docs
// TODO: make private
pub trait MutationExecutor<S, ExecutionStrategy> {
  fn execute_mutations(&self, solutions: &mut [S]);
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
  M: MutationOperator<S>,
{
  fn execute_mutations(&self, solutions: &mut [S]) {
    solutions.iter_mut().for_each(|s| self.mutate(s));
  }
}

impl<S, M> MutationExecutor<S, ParallelEachExecutionStrategy>
  for ParEach<MutationOperatorTag, S, M>
where
  S: Send + Sync,
  M: MutationOperator<S> + Sync,
{
  fn execute_mutations(&self, solutions: &mut [S]) {
    solutions
      .par_iter_mut()
      .for_each(|s| self.operator().mutate(s));
  }
}

impl<S, M> MutationExecutor<S, ParallelBatchExecutionStrategy>
  for ParBatch<MutationOperatorTag, S, M>
where
  S: Send + Sync,
  M: MutationOperator<S> + Sync,
{
  fn execute_mutations(&self, solutions: &mut [S]) {
    let chunk_size = (solutions.len() / rayon::current_num_threads()).max(1);
    solutions.par_chunks_mut(chunk_size).for_each(|chunk| {
      chunk.iter_mut().for_each(|s| self.operator().mutate(s))
    });
  }
}

/// `MutationOperator` that doesn't mutate given value.
pub struct NoMutation();

impl<S> MutationOperator<S> for NoMutation {
  fn mutate(&self, _: &mut S) {}
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn takes_mutator<ES, M: MutationExecutor<Solution, ES>>(m: &M) {
    m.execute_mutations(&mut []);
  }

  #[test]
  fn test_mutation_operator_from_closure() {
    let mutation_op = |solution: &mut Solution| *solution *= 2.0;
    takes_mutator(&mutation_op);
    takes_mutator(&mutation_op.par_each());
    takes_mutator(&mutation_op.par_batch());
  }

  #[test]
  fn test_mutator_from_closure() {
    let mutation = |solution: &mut Solution| *solution *= 2.0;
    takes_mutator(&mutation);
    takes_mutator(&mutation.par_each());
    takes_mutator(&mutation.par_batch());
  }

  #[test]
  fn test_custom_mutation() {
    struct CustomMutationOperator {}
    impl<S> MutationOperator<S> for CustomMutationOperator {
      fn mutate(&self, _: &mut S) {}
    }

    let mutation_op = CustomMutationOperator {};
    takes_mutator(&mutation_op);
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
