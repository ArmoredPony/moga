use crate::objective::Scores;

/// Terminates algorithm's execution based on some condition.
pub trait Terminator<'a, S: 'a, const N: usize> {
  /// Takes a slice of solutions and respective scores for each objective.
  /// If returns `true`, terminates algorithm's execution.
  fn terminate(
    &mut self,
    solutions_scores: impl Iterator<Item = (&'a S, &'a Scores<N>)>,
  ) -> bool;
}

impl<'a, const N: usize, S: 'a, F> Terminator<'a, S, N> for F
where
  S: Sync,
  F: Fn(&S, &Scores<N>) -> bool + Sync,
{
  fn terminate(
    &mut self,
    mut solutions_scores: impl Iterator<Item = (&'a S, &'a Scores<N>)>,
  ) -> bool {
    solutions_scores.any(|(sol, sc)| self(sol, sc))
  }
}

/// A `Terminator` that ignores solutions and terminates the algorithm when a
/// certain number of generations has passed.
pub struct GenerationCounter {
  generations: usize,
}

impl<'a, const N: usize, S: 'a> Terminator<'a, S, N> for GenerationCounter {
  fn terminate(
    &mut self,
    solutions_scores: impl Iterator<Item = (&'a S, &'a Scores<N>)>,
  ) -> bool {
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

  fn as_terminator<'a, const N: usize, T: Terminator<'a, Solution, N>>(_: &T) {}

  #[test]
  fn test_terminator_from_closure() {
    let t = |solution: &Solution, scores: &Scores<3>| {
      *solution > 0.0 && scores.iter().sum::<f32>() == 0.0
    };
    as_terminator(&t);
  }
}
