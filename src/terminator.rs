use rayon::prelude::*;

use crate::{
  execution::*,
  operator::{IntoPar, ParBatch, ParEach, TerminationOperatorTag},
  score::Scores,
};

/// TODO: add docs
pub trait TerminationCondition<S, const N: usize> {
  /// TODO: add docs
  fn terminate(&self, solution: &S, scores: &Scores<N>) -> bool;
}

impl<S, const N: usize, F> TerminationCondition<S, N> for F
where
  F: Fn(&S, &Scores<N>) -> bool,
{
  fn terminate(&self, solution: &S, scores: &Scores<N>) -> bool {
    self(solution, scores)
  }
}

impl<S, const N: usize, T> IntoPar<TerminationOperatorTag, S, N> for T where
  T: TerminationCondition<S, N>
{
}

/// Terminates algorithm's execution based on some condition.
pub trait Terminator<S, const N: usize> {
  /// Takes slices of solutions and their respective scores.
  /// If returns `true`, terminates algorithm's execution.
  fn terminate(&mut self, solutions: &[S], scores: &[Scores<N>]) -> bool;
}

// TODO: add docs
pub trait TerminatorExecutor<S, const N: usize, ExecutionStrategy> {
  fn terminate(&mut self, solutions: &[S], scores: &[Scores<N>]) -> bool;
}

impl<S, const N: usize, T> TerminatorExecutor<S, N, CustomExecutionStrategy>
  for T
where
  T: Terminator<S, N>,
{
  fn terminate(&mut self, solutions: &[S], scores: &[Scores<N>]) -> bool {
    Terminator::terminate(self, solutions, scores)
  }
}

impl<S, const N: usize, T> TerminatorExecutor<S, N, SequentialExecutionStrategy>
  for T
where
  T: TerminationCondition<S, N>,
{
  fn terminate(&mut self, solutions: &[S], scores: &[Scores<N>]) -> bool {
    solutions
      .iter()
      .zip(scores)
      .any(|(sol, sc)| TerminationCondition::terminate(self, sol, sc))
  }
}

impl<S, const N: usize, T>
  TerminatorExecutor<S, N, ParallelEachExecutionStrategy>
  for ParEach<TerminationOperatorTag, S, T>
where
  S: Sync,
  T: TerminationCondition<S, N> + Sync,
{
  fn terminate(&mut self, solutions: &[S], scores: &[Scores<N>]) -> bool {
    solutions
      .par_iter()
      .zip(scores)
      .any(|(sol, sc)| self.operator().terminate(sol, sc))
  }
}

impl<S, const N: usize, T>
  TerminatorExecutor<S, N, ParallelBatchExecutionStrategy>
  for ParBatch<TerminationOperatorTag, S, T>
where
  S: Sync,
  T: TerminationCondition<S, N> + Sync,
{
  fn terminate(&mut self, solutions: &[S], scores: &[Scores<N>]) -> bool {
    let chunk_size = (solutions.len() / rayon::current_num_threads()).max(1);
    solutions
      .chunks(chunk_size)
      .zip(scores.chunks(chunk_size))
      .par_bridge()
      .any(|chunk| {
        chunk
          .0
          .iter()
          .zip(chunk.1)
          .any(|(sol, sc)| self.operator().terminate(sol, sc))
      })
  }
}

/// A `Terminator` that ignores solutions and terminates the algorithm as soon
/// as a certain number of generations has passed.
pub struct GenerationsTerminator(pub usize);

impl<S, const N: usize> Terminator<S, N> for GenerationsTerminator {
  fn terminate(&mut self, _: &[S], _: &[Scores<N>]) -> bool {
    match self.0 {
      0 => true,
      _ => {
        self.0 -= 1;
        false
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn as_terminator<
    ES,
    const N: usize,
    T: TerminatorExecutor<Solution, N, ES>,
  >(
    t: &mut T,
  ) {
    t.terminate(&[], &[]);
  }

  #[test]
  fn test_terminator_from_closure() {
    let mut t = |solution: &Solution, scores: &Scores<3>| {
      *solution > 0.0 && scores.iter().sum::<f32>() == 0.0
    };
    as_terminator(&mut t);
    as_terminator(&mut t.par_each());
    as_terminator(&mut t.par_batch());
  }

  struct TerminateInstantly {}
  impl<S> Terminator<S, 3> for TerminateInstantly {
    fn terminate(&mut self, _: &[S], _: &[Scores<3>]) -> bool {
      true
    }
  }

  #[test]
  fn test_custom_terminator() {
    let mut t = TerminateInstantly {};
    as_terminator(&mut t);
  }
}
