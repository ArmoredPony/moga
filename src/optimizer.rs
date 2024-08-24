mod nsga;

/// Represents an abstract optimizer.
pub trait Optimizer<S> {
  /// Runs `Optimizer` until the termination condition is met. Returns
  /// `Solutions` from which all or only nondominated solutions can be
  /// extracted.
  fn run(self);
}

/// Contains solutions found by `Optimizer` from which all or only nondominated
/// solutions can be extracted.
pub trait Solutions<S> {
  // solutions: Vec<S>,
  // last_nondom_idx: usize,
}

// impl<S> Solutions<S> {
//   pub fn all_solutions(self) -> Vec<S> {
//     self.solutions
//   }

//   pub fn nondominated_solutions(self) -> Vec<S> {
//     let mut this = self;
//     this.solutions.truncate(this.last_nondom_idx);
//     this.solutions
//   }
// }
