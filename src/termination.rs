//! Termination operators and utilities.

use executor::TerminationExecutor;
use rayon::prelude::*;

use crate::{
  execution::strategy::*,
  operator::{
    tag::TerminationOperatorTag,
    ParBatch,
    ParBatchOperator,
    ParEach,
    ParEachOperator,
  },
  score::Scores,
};

/// An operator that for each given solution decides whether the algorithm
/// should be terminated or not.
///
/// Can be applied in parallel to each solution or to batches of solutions
/// by converting it into a parallelized operator with `par_each()` or
/// `par_batch()` methods.
///
/// # Examples
/// ```ignore
/// // stop if all fitness values are equal to zero or a solution became negative
/// let t = |f: &f32, v: &[f32; 3]| *f < 0.0 || v == &[0.0, 0.0, 0.0];
/// let t = t.par_batch();
/// ```
///
/// **Note that you always can implement this trait instead of using closures.**
pub trait Termination<S, const N: usize> {
  /// If returns `true`, the algorithm is terminated.
  fn terminate(&self, solution: &S, scores: &Scores<N>) -> bool;
}

impl<S, const N: usize, F> Termination<S, N> for F
where
  F: Fn(&S, &Scores<N>) -> bool,
{
  fn terminate(&self, solution: &S, scores: &Scores<N>) -> bool {
    self(solution, scores)
  }
}

impl<S, const N: usize, T> ParEach<TerminationOperatorTag, S, N, 0> for T
where
  S: Sync,
  T: Termination<S, N> + Sync,
{
}

impl<S, const N: usize, T> ParBatch<TerminationOperatorTag, S, N> for T
where
  S: Sync,
  T: Termination<S, N> + Sync,
{
}

/// An operator that terminates the algorithm based on some termination
/// condition.
///
/// # Examples
/// ```
/// let t = |fs: &[f32], _: &[[f32; 3]]| fs.iter().all(|f| *f < 1.0);
/// ```
///
/// **Note that you always can implement this trait instead of using closures.**
pub trait Terminator<S, const N: usize> {
  /// If returns `true`, the algorithm is terminated.
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

/// This private module prevents exposing the `Executor` to a user.
pub(crate) mod executor {
  use crate::score::Scores;

  /// An internal termination executor.
  pub trait TerminationExecutor<S, const N: usize, ExecutionStrategy> {
    /// Executes termination evaluation optionally parallelizing operator's
    /// application.
    fn execute_termination(
      &mut self,
      solutions: &[S],
      scores: &[Scores<N>],
    ) -> bool;
  }
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
  T: Termination<S, N>,
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
  for ParEachOperator<TerminationOperatorTag, S, T>
where
  S: Sync,
  T: Termination<S, N> + Sync,
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
  for ParBatchOperator<TerminationOperatorTag, S, T>
where
  S: Sync,
  T: Termination<S, N> + Sync,
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

/// A `Terminator` that terminates the algorithm as soon as a certain number of
/// generations have passed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GenerationTerminator(pub usize);

impl<S, const N: usize> Terminator<S, N> for GenerationTerminator {
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
  fn test_termination_from_closure() {
    let mut termination = |solution: &Solution, scores: &Scores<3>| {
      *solution > 0.0 && scores.iter().sum::<f32>() == 0.0
    };
    takes_terminator(&mut termination);
    takes_terminator(&mut termination.par_each());
    takes_terminator(&mut termination.par_batch());
  }

  #[test]
  fn test_terminatior_from_closure() {
    let mut terminator =
      |fs: &[f32], _: &[[f32; 3]]| fs.iter().all(|f| *f < 1.0);
    takes_terminator(&mut terminator);
  }

  #[test]
  fn test_custom_termination() {
    #[derive(Clone, Copy)]
    struct CustomTermination {}
    impl<S> Termination<S, 3> for CustomTermination {
      fn terminate(&self, _: &S, _: &Scores<3>) -> bool {
        true
      }
    }

    let mut termination = CustomTermination {};
    takes_terminator(&mut termination);
    takes_terminator(&mut termination.par_each());
    takes_terminator(&mut termination.par_batch());
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
