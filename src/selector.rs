use rand::prelude::*;
use rayon::prelude::*;

use crate::{execution::*, operator::*, score::Scores};

/// Decides if a solution should be selected as a parent for next population
/// or not.
pub trait SelectionOperator<S, const N: usize> {
  /// If returns true, then given solution will be selected as a parent for
  /// next population.
  fn select(&self, solution: &S, scores: &Scores<N>) -> bool;
}

impl<S, const N: usize, F> SelectionOperator<S, N> for F
where
  F: Fn(&S, &Scores<N>) -> bool,
{
  fn select(&self, solution: &S, scores: &Scores<N>) -> bool {
    self(solution, scores)
  }
}

impl<S, const N: usize, L> ParEach<SelectionOperatorTag, S, N, 0> for L where
  L: SelectionOperator<S, N>
{
}
impl<S, const N: usize, L> ParBatch<SelectionOperatorTag, S, N> for L where
  L: SelectionOperator<S, N>
{
}

/// Performs selection of suitable solutions to become parents for next
/// generation of solutions.
pub trait Selector<S, const N: usize> {
  /// Takes slices of solutions and their respective scores.
  /// Returns vector of references to selected solutions.
  fn select<'a>(&self, solutions: &'a [S], scores: &[Scores<N>]) -> Vec<&'a S>;
}

impl<S, const N: usize, F> Selector<S, N> for F
where
  F: for<'a> Fn(&'a [S], &[Scores<N>]) -> Vec<&'a S>,
{
  fn select<'a>(&self, solutions: &'a [S], scores: &[Scores<N>]) -> Vec<&'a S> {
    self(solutions, scores)
  }
}

// TODO: add docs
// TODO: make private
pub trait SelectionExecutor<S, const N: usize, ExecutionStrategy> {
  fn execute_selection<'a>(
    &self,
    solutions: &'a [S],
    scores: &[Scores<N>],
  ) -> Vec<&'a S>;
}

impl<S, const N: usize, L> SelectionExecutor<S, N, CustomExecutionStrategy>
  for L
where
  L: Selector<S, N>,
{
  fn execute_selection<'a>(
    &self,
    solutions: &'a [S],
    scores: &[Scores<N>],
  ) -> Vec<&'a S> {
    self.select(solutions, scores)
  }
}

impl<S, const N: usize, L> SelectionExecutor<S, N, SequentialExecutionStrategy>
  for L
where
  L: SelectionOperator<S, N>,
{
  fn execute_selection<'a>(
    &self,
    solutions: &'a [S],
    scores: &[Scores<N>],
  ) -> Vec<&'a S> {
    solutions
      .iter()
      .zip(scores)
      .filter_map(|(sol, sc)| self.select(sol, sc).then_some(sol))
      .collect()
  }
}

impl<S, const N: usize, L>
  SelectionExecutor<S, N, ParallelEachExecutionStrategy>
  for ParEachOperator<SelectionOperatorTag, S, L>
where
  S: Sync,
  L: SelectionOperator<S, N> + Sync,
{
  fn execute_selection<'a>(
    &self,
    solutions: &'a [S],
    scores: &[Scores<N>],
  ) -> Vec<&'a S> {
    solutions
      .par_iter()
      .zip(scores)
      .filter_map(|(sol, sc)| self.operator().select(sol, sc).then_some(sol))
      .collect()
  }
}

impl<S, const N: usize, L>
  SelectionExecutor<S, N, ParallelBatchExecutionStrategy>
  for ParBatchOperator<SelectionOperatorTag, S, L>
where
  S: Sync,
  L: SelectionOperator<S, N> + Sync,
{
  fn execute_selection<'a>(
    &self,
    solutions: &'a [S],
    scores: &[Scores<N>],
  ) -> Vec<&'a S> {
    let chunk_size = (solutions.len() / rayon::current_num_threads()).max(1);
    solutions
      .chunks(chunk_size)
      .zip(scores.chunks(chunk_size))
      .par_bridge()
      .flat_map_iter(|chunk| {
        chunk.0.iter().zip(chunk.1).filter_map(|(sol, sc)| {
          self.operator().select(sol, sc).then_some(sol)
        })
      })
      .collect()
  }
}

/// Selects all solutions. No discrimination whatsoever.
pub struct AllSelector();

impl<const N: usize, S> Selector<S, N> for AllSelector {
  fn select<'a>(&self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().collect()
  }
}

/// Selects `n` first solutions. 'First' doesn't mean the best, this selector
/// just returns `n` solutions it sees first.
pub struct FirstSelector(pub usize);

impl<const N: usize, S> Selector<S, N> for FirstSelector {
  fn select<'a>(&self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().take(self.0).collect()
  }
}

/// Selects `n` random solutions.
pub struct RandomSelector(pub usize);

impl<const N: usize, S> Selector<S, N> for RandomSelector {
  fn select<'a>(&self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions
      .iter()
      .choose_multiple(&mut rand::thread_rng(), self.0)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn takes_selector<
    ES,
    const N: usize,
    L: SelectionExecutor<Solution, N, ES>,
  >(
    l: &mut L,
  ) {
    l.execute_selection(&[], &[]);
  }

  #[test]
  fn test_selection_operator_from_closure() {
    let mut selection_op = |_: &Solution, _: &Scores<0>| true;
    takes_selector(&mut selection_op);
  }

  #[test]
  fn test_selector_from_fn() {
    fn selector<'a>(
      solutions: &'a [Solution],
      _: &[Scores<3>],
    ) -> Vec<&'a Solution> {
      solutions.iter().collect()
    }
    takes_selector(&mut &selector);
  }

  // will work once `#![feature(closure_lifetime_binder)]`
  // is fucking stabilized already it's been two years since it's implemented

  // #[test]
  // fn test_selector_from_closure() {
  //   let select_all = for<'a, 'b>
  //    |solutions: &'a [Solution], _: &'b [Scores<3>]| -> Vec<&'a Solution> {
  //      solutions.iter().collect::<Vec<&'a Solution>>()
  //     };
  //   takes_selector(&select_all);
  // }

  #[test]
  fn test_custom_selection_operator() {
    struct CustomSelectionOperator {}
    impl<S, const N: usize> SelectionOperator<S, N> for CustomSelectionOperator {
      fn select(&self, _: &S, _: &Scores<N>) -> bool {
        true
      }
    }

    let mut selection_op = CustomSelectionOperator {};
    takes_selector::<SequentialExecutionStrategy, 0, CustomSelectionOperator>(
      &mut selection_op,
    );
  }

  #[test]
  fn test_custom_selectior() {
    struct CustomSelector {}
    impl<S, const N: usize> Selector<S, N> for CustomSelector {
      fn select<'a>(&self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
        solutions.iter().collect()
      }
    }

    let mut selector = CustomSelector {};
    takes_selector::<CustomExecutionStrategy, 0, CustomSelector>(&mut selector);
  }

  #[test]
  fn test_all_selector() {
    let mut selector = AllSelector();
    takes_selector::<CustomExecutionStrategy, 0, AllSelector>(&mut selector);
  }

  #[test]
  fn test_first_selector() {
    let mut selector = FirstSelector(10);
    takes_selector::<CustomExecutionStrategy, 0, FirstSelector>(&mut selector);
  }

  #[test]
  fn test_random_selector() {
    let mut selector = RandomSelector(10);
    takes_selector::<CustomExecutionStrategy, 0, RandomSelector>(&mut selector);
  }
}
