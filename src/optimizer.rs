pub mod nsga;

pub use nsga::*;

/// Represents an abstract optimizer.
pub trait Optimizer<S> {
  /// Runs `Optimizer` until the termination condition is met. Returns
  /// `Solutions` from which all or only nondominated solutions can be
  /// extracted.
  fn run(self) -> Vec<S>;
}
