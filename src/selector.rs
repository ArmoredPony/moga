use rand::prelude::*;

use crate::objective::Scores;

/// Performs selection of suitable solutions.
pub trait Selector<S, const N: usize> {
  /// Takes slices of solutions and their respective scores.
  /// Returns vector of selected solutions.
  fn select<'a>(
    &mut self,
    solutions: &'a [S],
    scores: &[Scores<N>],
  ) -> Vec<&'a S>;
}
/// Selects all solutions. No discrimination whatsoever.
pub struct AllSelector();

impl<const N: usize, S> Selector<S, N> for AllSelector {
  fn select<'a>(&mut self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().collect()
  }
}
/// Selects `n` first solutions. 'First' doesn't mean the best, this selector
/// just returns `n` solutions it sees first.
pub struct FirstSelector(pub usize);

impl<const N: usize, S> Selector<S, N> for FirstSelector {
  fn select<'a>(&mut self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().take(self.0).collect()
  }
}
/// Selects `n` random solutions. You may provide any type that implements
/// `Rng` trait from [rand](https://crates.io/crates/rand) crate.
pub struct RandomSelector<R: Rng>(pub usize, pub R);

impl<const N: usize, S, R: Rng> Selector<S, N> for RandomSelector<R> {
  fn select<'a>(&mut self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().choose_multiple(&mut self.1, self.0)
  }
}

impl<const N: usize, S, F> Selector<S, N> for F
where
  F: for<'a> Fn(&'a [S], &[Scores<N>]) -> Vec<&'a S>,
{
  fn select<'a>(
    &mut self,
    solutions: &'a [S],
    scores: &[Scores<N>],
  ) -> Vec<&'a S> {
    self(solutions, scores)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn as_selector<const N: usize, S: Selector<Solution, N>>(_: &S) {}

  #[test]
  fn test_selector_from_fn() {
    fn select_all<'a>(
      solutions: &'a [Solution],
      _: &[Scores<3>],
    ) -> Vec<&'a Solution> {
      solutions.iter().collect()
    }
    as_selector(&select_all);
  }

  #[test]
  fn test_selector_from_closure() {
    // will work once `#![feature(closure_lifetime_binder)]`
    // is fucking stabilized already it's been two years since it's implemented

    // let select_all = for<'a, 'b>
    //  |solutions: &'a [Solution], _: &'b [Scores<3>]| -> Vec<&'a Solution> {
    //    solutions.iter().collect::<Vec<&'a Solution>>()
    //   };
    // as_selector(&select_all);
  }

  #[test]
  fn test_all_selector() {
    let s = AllSelector();
    as_selector::<0, AllSelector>(&s);
  }

  #[test]
  fn test_first_selector() {
    let s = FirstSelector(10);
    as_selector::<0, FirstSelector>(&s);
  }

  #[test]
  fn test_random_selector() {
    let s = RandomSelector(10, rand::thread_rng());
    as_selector::<0, RandomSelector<_>>(&s);
  }
}
