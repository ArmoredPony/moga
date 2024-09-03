//! Abstract optimizer.

mod genetic_algorithm;
pub mod nsga;

/// Represents an abstract optimizer.
pub trait Optimizer<Solution, const OBJECTIVE_NUM: usize>: Sized {
  /// Runs `Optimizer` until the termination condition is met, then returns
  /// the last found population.
  fn optimize(self) -> Vec<Solution>;
}
