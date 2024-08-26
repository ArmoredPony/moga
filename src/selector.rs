use rand::{prelude::*, rngs::SmallRng};

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
// TODO: add docs
pub struct AllSelector();

impl<const N: usize, S> Selector<S, N> for AllSelector {
  fn select<'a>(&mut self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().collect()
  }
}
// TODO: add docs
pub struct FirstSelector(usize);

impl<const N: usize, S> Selector<S, N> for FirstSelector {
  fn select<'a>(&mut self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().take(self.0).collect()
  }
}
// TODO: add docs
pub struct RandomSelector(usize, SmallRng);

impl<const N: usize, S> Selector<S, N> for RandomSelector {
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
  fn test_selector_from_closure<'a>() {
    let select_all = |solutions: &'a [Solution], _: &[Scores<3>]| {
      solutions.iter().collect::<Vec<&'a Solution>>()
    };
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
    let s = RandomSelector(10, SmallRng::from_entropy());
    as_selector::<0, RandomSelector>(&s);
  }
}
