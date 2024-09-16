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
  score::{ParetoDominance, Scores},
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

/// Selects at most `n` first solutions. 'First' doesn't mean the best, this
/// selector just returns `n` solutions it meets first.
///
/// If `n` is bigger than the number of solutions, this selector selects all
/// solutions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FirstSelector(pub usize);

impl<const N: usize, S> Selector<S, N> for FirstSelector {
  fn select<'a>(&self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().take(self.0).collect()
  }
}

/// Selects at most `n` random solutions.
///
/// If `n` is bigger than the number of solutions, this selector selects all
/// solutions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RandomSelector(pub usize);

impl<const N: usize, S> Selector<S, N> for RandomSelector {
  fn select<'a>(&self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions
      .choose_multiple(&mut rand::thread_rng(), self.0)
      .collect()
  }
}

/// Selects at most `n` solutions from random chunks of *unique* solutions of
/// size `k`. Each solution can be selected only once.
///
/// From each chunk, the least dominated solution is selected. If there are
/// multiple equally dominated solutions, a random one is selected.
/// This selector selects one solution per a chunk, so it may select less than
/// `n` solutions. If you want to select all solutions this selector can proide,
/// set this value to `usize::MAX`.
///
/// # Panics
///
/// Will panic in runtime if `k` is 0.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TournamentSelectorWithoutReplacement(pub usize, pub usize);

impl<const N: usize, S> Selector<S, N>
  for TournamentSelectorWithoutReplacement
{
  fn select<'a>(&self, solutions: &'a [S], scores: &[Scores<N>]) -> Vec<&'a S> {
    rand::seq::index::sample(
      &mut rand::thread_rng(),
      solutions.len(),
      solutions.len(),
    )
    .into_vec()
    .chunks(self.1)
    .take(self.0)
    .map(|chunk| {
      chunk
        .iter()
        .min_by(|i, j| scores[**i].dominance(&scores[**j]))
        .map(|idx| &solutions[*idx])
        .expect("chunk must not be empty")
    })
    .collect()
  }
}

/// Selects `n` solutions from random chunks of solutions of size `k`.
/// Each solution can be selected multiple times.
///
/// From each chunk, the least dominated solution is selected. If there are
/// multiple equally dominated solutions, a random one is selected. If `k` is
/// bigger than the number of solutions, all solutions will form a single chunk
/// from which `n` solutions will be selected.
///
/// # Panics
///
/// Will panic in runtime if `k` is 0.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TournamentSelectorWithReplacement(pub usize, pub usize);

impl<const N: usize, S> Selector<S, N> for TournamentSelectorWithReplacement {
  fn select<'a>(&self, solutions: &'a [S], scores: &[Scores<N>]) -> Vec<&'a S> {
    (0..self.0)
      .map(|_| {
        rand::seq::index::sample(
          &mut rand::thread_rng(),
          solutions.len(),
          self.1.min(solutions.len()),
        )
        .iter()
        .min_by(|i, j| scores[*i].dominance(&scores[*j]))
        .map(|idx| &solutions[idx])
        .expect("chunk must not be empty")
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
}
