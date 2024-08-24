use crate::objective::Scores;

/// Performs selection of suitable solutions.
pub trait Selector<const N: usize, S> {
  /// Takes a slice of solutions and their scores and returns vector of
  /// selected solutions.
  fn select<'a>(&self, solutions_scores: &[(&'a S, &Scores<N>)]) -> Vec<&'a S>;
}

struct SelectAll();

impl<const N: usize, S> Selector<N, S> for SelectAll {
  fn select<'a>(&self, solutions_scores: &[(&'a S, &Scores<N>)]) -> Vec<&'a S> {
    solutions_scores.iter().map(|(sol, _)| *sol).collect()
  }
}

pub fn select_all<const N: usize, S>() -> impl Selector<N, S> {
  SelectAll()
}

struct SelectFirst(usize);

impl<const N: usize, S> Selector<N, S> for SelectFirst {
  fn select<'a>(&self, solutions_scores: &[(&'a S, &Scores<N>)]) -> Vec<&'a S> {
    solutions_scores
      .iter()
      .take(self.0)
      .map(|(sol, _)| *sol)
      .collect()
  }
}

pub fn select_first<const N: usize, S>(n: usize) -> impl Selector<N, S> {
  SelectFirst(n)
}

impl<const N: usize, S, F> Selector<N, S> for F
where
  F: for<'a> Fn(&[(&'a S, &Scores<N>)]) -> Vec<&'a S>,
{
  fn select<'a>(&self, solutions: &[(&'a S, &Scores<N>)]) -> Vec<&'a S> {
    self(solutions)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn as_selector<const N: usize, S: Selector<N, Solution>>(_: &S) {}

  #[test]
  fn test_selector_from_fn() {
    fn select_all<'a>(
      solutions_scores: &[(&'a Solution, &Scores<3>)],
    ) -> Vec<&'a Solution> {
      solutions_scores.iter().map(|(sol, _)| *sol).collect()
    }
    as_selector(&select_all);
  }

  #[test]
  fn test_selector_all() {
    let s = select_all::<3, Solution>();
    as_selector(&s);
  }

  #[test]
  fn test_selector_first() {
    let s = select_first::<3, Solution>(10);
    as_selector(&s);
  }
}
