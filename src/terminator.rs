use crate::objective::Scores;

/// Terminates algorithm's execution based on some condition.
pub trait Terminator<S, const N: usize> {
  /// Takes slices of solutions and their respective scores.
  /// If returns `true`, terminates algorithm's execution.
  fn terminate(&mut self, solutions: &[S], scores: &[Scores<N>]) -> bool;
}

impl<const N: usize, S, F> Terminator<S, N> for F
where
  S: Sync,
  F: Fn(&S, &Scores<N>) -> bool + Sync,
{
  fn terminate(&mut self, solutions: &[S], scores: &[Scores<N>]) -> bool {
    solutions.iter().zip(scores).any(|(sol, sc)| self(sol, sc))
  }
}

/// `Terminator` that ignores solutions and terminates the algorithm as soon
/// as a certain number of generations has passed.
pub struct GenerationsTerminator(usize); // TODO: add tests

impl<const N: usize, S> Terminator<S, N> for GenerationsTerminator {
  fn terminate(&mut self, _: &[S], _: &[Scores<N>]) -> bool {
    match self.0 {
      0 => true,
      _ => {
        self.0 -= 1;
        false
      }
    }
  }
}

/// `Terminator` that returns true if exists at least one solution, which
/// scores values are less than or equal to respective target scores values.
pub struct ScoresTerminator<const N: usize>(Scores<N>); // TODO: add tests

impl<const N: usize, S> Terminator<S, N> for ScoresTerminator<N> {
  fn terminate(&mut self, _: &[S], scores: &[Scores<N>]) -> bool {
    scores
      .iter()
      .any(|s| s.iter().zip(self.0.iter()).all(|(&a, &b)| a <= b))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f32;

  fn as_terminator<const N: usize, T: Terminator<Solution, N>>(_: &T) {}

  #[test]
  fn test_terminator_from_closure() {
    let t = |solution: &Solution, scores: &Scores<3>| {
      *solution > 0.0 && scores.iter().sum::<f32>() == 0.0
    };
    as_terminator(&t);
  }
}
