//! Abstract optimizer.

use std::{error::Error, fmt::Display};

pub mod nsga;
pub mod spea;

/// Represents an abstract optimizer.
pub trait Optimizer<Solution, const OBJECTIVE_NUM: usize>: Sized {
  /// Runs `Optimizer` until the termination condition is met, then returns
  /// the last found population or an optimization error.
  fn optimize(self) -> Result<Vec<Solution>, OptimizationError>;
}

/// Optimization failure reasons.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OptimizationError {
  /// There are no solutions in the current population.
  PopulationEmpty,
  /// Number of calculated scores is not equal to number of solutions.
  ScoreCountMismatch {
    /// Actual number of scores.
    actual: usize,
    /// Expected number of scores.
    expected: usize,
  },
}

impl Display for OptimizationError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::PopulationEmpty => write!(f, "population is empty"),
      Self::ScoreCountMismatch { actual, expected } => {
        write!(f, "expected {expected} scores, got {actual}")
      }
    }
  }
}

impl Error for OptimizationError {}
