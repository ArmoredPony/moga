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

pub fn select_all<S, const N: usize>() -> impl Selector<S, N> {
  SelectAll()
}

pub fn select_first<S, const N: usize>(n: usize) -> impl Selector<S, N> {
  SelectFirst(n)
}

pub fn select_random<S, const N: usize>(n: usize) -> impl Selector<S, N> {
  SelectRandom(n, SmallRng::from_entropy())
}

struct SelectAll();

impl<const N: usize, S> Selector<S, N> for SelectAll {
  fn select<'a>(&mut self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().collect()
  }
}

struct SelectFirst(usize);

impl<const N: usize, S> Selector<S, N> for SelectFirst {
  fn select<'a>(&mut self, solutions: &'a [S], _: &[Scores<N>]) -> Vec<&'a S> {
    solutions.iter().take(self.0).collect()
  }
}

struct SelectRandom(usize, SmallRng);

impl<const N: usize, S> Selector<S, N> for SelectRandom {
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
  fn test_selector_all() {
    let s = select_all::<Solution, 3>();
    as_selector(&s);
  }

  #[test]
  fn test_selector_first() {
    let s = select_first::<Solution, 3>(10);
    as_selector(&s);
  }

  #[test]
  fn test_select_random() {
    let s = select_random::<Solution, 3>(10);
    as_selector(&s);
  }
}
