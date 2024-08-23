use rayon::prelude::*;

/// Terminates algorithm's execution based on some condition.
pub trait Terminator<S> {
  /// Takes a slice of solutions. If returns `true`, terminates algorithm's
  /// execution.
  fn terminate(&mut self, solutions: &[S]) -> bool;
}

impl<S, F> Terminator<S> for F
where
  S: Sync,
  F: Fn(&S) -> bool + Sync,
{
  fn terminate(&mut self, solutions: &[S]) -> bool {
    solutions.par_iter().any(self as &F)
  }
}

/// A `Terminator` that ignores solutions and terminates the algorithm when a
/// certain number of generations has passed.
pub struct GenerationCounter {
  generations: usize,
}

impl GenerationCounter {
  pub fn new(generations: usize) -> Self {
    Self { generations }
  }
}

impl<S> Terminator<S> for GenerationCounter {
  fn terminate(&mut self, _: &[S]) -> bool {
    match self.generations {
      0 => true,
      _ => {
        self.generations -= 1;
        false
      }
    }
  }
}
