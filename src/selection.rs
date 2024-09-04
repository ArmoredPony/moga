//! Selection operators and utilities.

use executor::SelectionExecutor;
use rand::prelude::*;
use rayon::prelude::*;

use crate::{
  execution::strategy::*,
  operator::{
    tag::SelectionOperatorTag,
    ParBatch,
    ParBatchOperator,
    ParEach,
    ParEachOperator,
  },
  score::Scores,
  Score,
};

/// An operator that decides whether a solution will be selected as a parent
/// for the next generation of solutions or not. Selected solutions' references
/// are passed into `Recombinator`.
///
/// Can be applied in parallel to each solution or to batches of solutions
/// by converting it into a parallelized operator with `par_each()` or
/// `par_batch()` methods.
///
/// # Examples
/// ```ignore
/// let s = |_: &f32, _: &[f32; 3]| true; // simply selects each solution
/// let s = s.par_each();
/// ```
///
/// **Note that you always can implement this trait instead of using closures.**
pub trait Selection<S, const N: usize> {
  /// If returns true, then given solution will be selected as a parent for
  /// next population.
  fn select(&self, solution: &S, scores: &Scores<N>) -> bool;
}

impl<S, const N: usize, F> Selection<S, N> for F
where
  F: Fn(&S, &Scores<N>) -> bool,
{
  fn select(&self, solution: &S, scores: &Scores<N>) -> bool {
    self(solution, scores)
  }
}

impl<S, const N: usize, L> ParEach<SelectionOperatorTag, S, N, 0> for L where
  L: Selection<S, N>
{
}
impl<S, const N: usize, L> ParBatch<SelectionOperatorTag, S, N> for L where
  L: Selection<S, N>
{
}

/// An operator that selects solutions suitable for recombination into a new
/// generation of solutions. Selected solutions' references are passed into
/// `Recombinator`.
///
/// Due to the fact that [closure lifetime binders] are still unimplemented
/// (and [it doesn't feel like] they are going to be implemented soon),
/// `Selector`s in closure form are a pain to work with. In fact, you can only
/// implement them using a `fn` function with a lifetime parameter:
/// ```
/// // selects all solutions for recombination
/// fn selector<'a>(fs: &'a [f32], _: &[[f32; 3]]) -> Vec<&'a f32> {
///   fs.iter().collect()
/// }
/// ```
///
/// To save you the trouble, this crate provides several `Selector`s
/// implementations such as `AllSelector`, `RandomSelector`, etc. And if you
/// want to create your own selector after all, consider implementing `Selector`
/// trait.
///
/// **Note that you probably want to implement this trait instead of using closures.**
///
/// [closure lifetime binders]: https://rust-lang.github.io/rfcs/3216-closure-lifetime-binder.html
/// [it doesn't feel like]: https://github.com/rust-lang/rust/issues/97362
pub trait Selector<S, const N: usize> {
  /// Takes slices of solutions and their respective scores.
  /// Returns a vector of references to selected solutions.
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

/// This private module prevents exposing the `Executor` to a user.
pub(crate) mod executor {
  use crate::score::Scores;

  /// An internal selecion executor.
  pub trait SelectionExecutor<S, const N: usize, ExecutionStrategy> {
    /// Executes selection optionally parallelizing operator's application.
    fn execute_selection<'a>(
      &self,
      solutions: &'a [S],
      scores: &[Scores<N>],
    ) -> Vec<&'a S>;
  }
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
  L: Selection<S, N>,
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
  L: Selection<S, N> + Sync,
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
  L: Selection<S, N> + Sync,
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
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct AllSelector();

impl<const N: usize, S> Selector<S, N> for AllSelector {
  fn select<'a>(&self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().collect()
  }
}

/// Selects `n` first solutions. 'First' doesn't mean the best, this selector
/// just returns `n` solutions it sees first.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FirstSelector(pub usize);

impl<const N: usize, S> Selector<S, N> for FirstSelector {
  fn select<'a>(&self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().take(self.0).collect()
  }
}

/// Selects `n` random solutions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RandomSelector(pub usize);

impl<const N: usize, S> Selector<S, N> for RandomSelector {
  fn select<'a>(&self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions
      .iter()
      .choose_multiple(&mut rand::thread_rng(), self.0)
  }
}

/// Selects `n` solutions with smallest sum of fitness scores.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BestSelector(pub usize);

impl<const N: usize, S> Selector<S, N> for BestSelector {
  fn select<'a>(&self, solutions: &'a [S], scores: &[Scores<N>]) -> Vec<&'a S> {
    if solutions.len() <= self.0 {
      return solutions.iter().collect();
    }
    let mut sol_sc = solutions
      .iter()
      .zip(
        scores
          .iter()
          .map(|sc| sc.map(Score::abs).iter().sum::<Score>()),
      )
      .collect::<Vec<_>>();
    sol_sc.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("NaN encountered"));
    sol_sc.into_iter().take(self.0).map(|s| s.0).collect()
  }
}

/// Splits solutions into chunks of size `n` and selects the best solution from
/// each chunk.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TournamentSelector(pub usize);

impl<const N: usize, S> Selector<S, N> for TournamentSelector {
  fn select<'a>(&self, solutions: &'a [S], scores: &[Scores<N>]) -> Vec<&'a S> {
    let mut sol_sc = solutions
      .iter()
      .zip(
        scores
          .iter()
          .map(|sc| sc.map(Score::abs).iter().sum::<Score>()),
      )
      .collect::<Vec<_>>();
    sol_sc.shuffle(&mut rand::thread_rng());
    sol_sc
      .chunks(self.0)
      .filter_map(|chunk| {
        chunk
          .iter()
          .min_by(|a, b| a.1.partial_cmp(&b.1).expect("NaN encoutnered"))
          .map(|ch| ch.0)
      })
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn takes_selector<ES, L: SelectionExecutor<Solution, 3, ES>>(l: &L) {
    l.execute_selection(&[], &[]);
  }

  #[test]
  fn test_selection_from_closure() {
    let selection = |_: &Solution, _: &Scores<3>| true;
    takes_selector(&selection);
    takes_selector(&selection.par_each());
    takes_selector(&selection.par_batch());
  }

  #[test]
  fn test_selector_from_fn() {
    fn selector<'a>(
      solutions: &'a [Solution],
      _: &[Scores<3>],
    ) -> Vec<&'a Solution> {
      solutions.iter().collect()
    }
    takes_selector(&selector);
  }

  // will work once `#![feature(closure_lifetime_binder)]`is stabilized already
  // it's been two years since it's implemented god damn it

  // #[test]
  // fn test_selector_from_closure() {
  //   let select_all = for<'a, 'b>
  //    |solutions: &'a [Solution], _: &'b [Scores<3>]| -> Vec<&'a Solution> {
  //      solutions.iter().collect::<Vec<&'a Solution>>()
  //     };
  //   takes_selector(&select_all);
  // }

  #[test]
  fn test_custom_selection() {
    #[derive(Clone, Copy)]
    struct CustomSelection {}
    impl<S> Selection<S, 3> for CustomSelection {
      fn select(&self, _: &S, _: &Scores<3>) -> bool {
        true
      }
    }

    let selection = CustomSelection {};
    takes_selector(&selection);
    takes_selector(&selection.par_each());
    takes_selector(&selection.par_batch());
  }

  #[test]
  fn test_custom_selectior() {
    #[derive(Clone, Copy)]
    struct CustomSelector {}
    impl<S> Selector<S, 3> for CustomSelector {
      fn select<'a>(&self, solutions: &'a [S], _: &[Scores<3>]) -> Vec<&'a S> {
        solutions.iter().collect()
      }
    }

    let selector = CustomSelector {};
    takes_selector(&selector);
  }

  #[test]
  fn test_all_selector() {
    let selector = AllSelector();
    takes_selector(&selector);
  }

  #[test]
  fn test_first_selector() {
    let selector = FirstSelector(10);
    takes_selector(&selector);
  }

  #[test]
  fn test_random_selector() {
    let selector = RandomSelector(10);
    takes_selector(&selector);
  }

  #[test]
  fn test_best_selector() {
    let selector = BestSelector(10);
    takes_selector(&selector);
  }

  #[test]
  fn test_tournament_selector() {
    let selector = TournamentSelector(10);
    takes_selector(&selector);
  }
}
