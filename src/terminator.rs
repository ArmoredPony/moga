use rayon::prelude::*;

use crate::objective::Scores;

/// Terminates algorithm's execution based on some condition.
pub trait Terminator<const N: usize, S> {
  /// Takes a slice of solutions and respective scores for each objective.
  /// If returns `true`, terminates algorithm's execution.
  fn terminate(&mut self, solutions_scores: &[(S, Scores<N>)]) -> bool;
}

impl<const N: usize, S, F> Terminator<N, S> for F
where
  S: Sync,
  F: Fn(&S, &Scores<N>) -> bool + Sync,
{
  fn terminate(&mut self, solutions_scores: &[(S, Scores<N>)]) -> bool {
    solutions_scores.par_iter().any(|(sol, sc)| self(sol, sc))
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

impl<const N: usize, S> Terminator<N, S> for GenerationCounter {
  fn terminate(&mut self, _: &[(S, Scores<N>)]) -> bool {
    match self.generations {
      0 => true,
      _ => {
        self.generations -= 1;
        false
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn as_terminator<const N: usize, T: Terminator<N, Solution>>(_: &T) {}

  #[test]
  fn test_terminator_from_closure() {
    let t = |solution: &Solution, scores: &Scores<3>| {
      *solution > 0.0 && scores.iter().sum::<f32>() == 0.0
    };
    as_terminator(&t);
  }
}
