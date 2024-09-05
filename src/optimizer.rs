//! Abstract optimizer.

pub mod nsga;
pub mod spea;

/// Represents an abstract optimizer.
pub trait Optimizer<Solution, const OBJECTIVE_NUM: usize>: Sized {
  /// Runs `Optimizer` until the termination condition is met, then returns
  /// the last found population.
  fn optimize(self) -> Vec<Solution>;
}
