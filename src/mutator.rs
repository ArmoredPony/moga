use rayon::prelude::*;

use crate::{execution::*, operator::*};

/// Mutates a solution.
pub trait Mutation<S> {
  /// Takes a solution and mutates it.
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

impl<S, M> ParEach<MutationOperatorTag, S, 0, 0> for M where M: Mutation<S> {}

impl<S, M> ParBatch<MutationOperatorTag, S, 0> for M where M: Mutation<S> {}

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
  M: Mutation<S>,
{
  fn execute_mutations(&self, solutions: &mut [S]) {
    solutions.iter_mut().for_each(|s| self.mutate(s));
  }
}

impl<S, M> MutationExecutor<S, ParallelEachExecutionStrategy>
  for ParEachOperator<MutationOperatorTag, S, M>
where
  S: Send + Sync,
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
  S: Send + Sync,
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
    let mutation = |solution: &mut Solution| *solution *= 2.0;
    takes_mutator(&mutation);
    takes_mutator(&mutation.par_each());
    takes_mutator(&mutation.par_batch());
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
