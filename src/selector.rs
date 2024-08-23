use rayon::prelude::*;

/// Performs selection of suitable solutions.
pub trait Selector<S> {
  /// Takes a slice of solutions and returns vector of selected solutions.
  fn select<'a>(&self, solutions: &'a [S]) -> Vec<&'a S>;
}

impl<S, F> Selector<S> for F
where
  S: Sync,
  F: Fn(&S) -> bool + Sync,
{
  fn select<'a>(&self, solutions: &'a [S]) -> Vec<&'a S> {
    solutions.par_iter().filter(|s| self(s)).collect()
  }
}
