use rayon::prelude::*;

use crate::{
  execution::*,
  operator::{IntoParOperator, ParBatch, ParEach, TerminationOperatorTag},
  score::Scores,
};

/// TODO: add docs
pub trait TerminationOperator<S, const N: usize> {
  /// TODO: add docs
  fn terminate(&self, solution: &S, scores: &Scores<N>) -> bool;
}

impl<S, const N: usize, F> TerminationOperator<S, N> for F
where
  F: Fn(&S, &Scores<N>) -> bool,
{
  fn terminate(&self, solution: &S, scores: &Scores<N>) -> bool {
    self(solution, scores)
  }
}

impl<S, const N: usize, T> IntoParOperator<TerminationOperatorTag, S, N> for T where
  T: TerminationOperator<S, N>
{
}

/// Terminates algorithm's execution based on some condition.
pub trait Terminator<S, const N: usize> {
  /// Takes slices of solutions and their respective scores.
  /// If returns `true`, terminates algorithm's execution.
  fn terminate(&mut self, solutions: &[S], scores: &[Scores<N>]) -> bool;
}

impl<S, const N: usize, F> Terminator<S, N> for F
where
  F: FnMut(&[S], &[Scores<N>]) -> bool,
{
  fn terminate(&mut self, solutions: &[S], scores: &[Scores<N>]) -> bool {
    self(solutions, scores)
  }
}

// TODO: add docs
// TODO: make private
pub trait TerminationExecutor<S, const N: usize, ExecutionStrategy> {
  fn execute_termination(
    &mut self,
    solutions: &[S],
    scores: &[Scores<N>],
  ) -> bool;
}

impl<S, const N: usize, T> TerminationExecutor<S, N, CustomExecutionStrategy>
  for T
where
  T: Terminator<S, N>,
{
  fn execute_termination(
    &mut self,
    solutions: &[S],
    scores: &[Scores<N>],
  ) -> bool {
    self.terminate(solutions, scores)
  }
}

impl<S, const N: usize, T>
  TerminationExecutor<S, N, SequentialExecutionStrategy> for T
where
  T: TerminationOperator<S, N>,
{
  fn execute_termination(
    &mut self,
    solutions: &[S],
    scores: &[Scores<N>],
  ) -> bool {
    solutions
      .iter()
      .zip(scores)
      .any(|(sol, sc)| self.terminate(sol, sc))
  }
}

impl<S, const N: usize, T>
  TerminationExecutor<S, N, ParallelEachExecutionStrategy>
  for ParEach<TerminationOperatorTag, S, T>
where
  S: Sync,
  T: TerminationOperator<S, N> + Sync,
{
  fn execute_termination(
    &mut self,
    solutions: &[S],
    scores: &[Scores<N>],
  ) -> bool {
    solutions
      .par_iter()
      .zip(scores)
      .any(|(sol, sc)| self.operator().terminate(sol, sc))
  }
}

impl<S, const N: usize, T>
  TerminationExecutor<S, N, ParallelBatchExecutionStrategy>
  for ParBatch<TerminationOperatorTag, S, T>
where
  S: Sync,
  T: TerminationOperator<S, N> + Sync,
{
  fn execute_termination(
    &mut self,
    solutions: &[S],
    scores: &[Scores<N>],
  ) -> bool {
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
/// as a certain number of generations had passed.
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

  fn takes_terminator<
    ES,
    const N: usize,
    T: TerminationExecutor<Solution, N, ES>,
  >(
    t: &mut T,
  ) {
    t.execute_termination(&[], &[]);
  }

  #[test]
  fn test_termination_operator_from_closure() {
    let mut termination_op = |solution: &Solution, scores: &Scores<3>| {
      *solution > 0.0 && scores.iter().sum::<f32>() == 0.0
    };
    takes_terminator(&mut termination_op);
    takes_terminator(&mut termination_op.par_each());
    takes_terminator(&mut termination_op.par_batch());
  }

  #[test]
  fn test_terminatior_from_closure() {
    let mut terminator = |solutions: &[Solution], scores: &[Scores<3>]| {
      solutions.iter().any(|s| *s > 0.0) || scores.iter().any(|s| s[0] == 0.0)
    };
    takes_terminator(&mut terminator);
  }

  #[test]
  fn test_custom_termination_operator() {
    struct CustomTerminationOperator {}
    impl<S> TerminationOperator<S, 3> for CustomTerminationOperator {
      fn terminate(&self, _: &S, _: &Scores<3>) -> bool {
        true
      }
    }

    let mut termination_op = CustomTerminationOperator {};
    takes_terminator(&mut termination_op);
  }

  #[test]
  fn test_custom_terminator() {
    struct CustomTerminator {}
    impl<S> Terminator<S, 3> for CustomTerminator {
      fn terminate(&mut self, _: &[S], _: &[Scores<3>]) -> bool {
        true
      }
    }

    let mut terminator = CustomTerminator {};
    takes_terminator(&mut terminator);
  }
}
