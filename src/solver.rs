mod nsga;
mod pareto;

use std::cmp::Ordering;

use crate::{
  crossover::Crossover, mutator::Mutator, objective::Objectives,
  selector::Selector, terminator::Terminator,
};

/// Represents an abstract problem solver.
pub trait Solver<
  S,
  const OBJ_CNT: usize,
  const CRS_IN: usize,
  const CRS_OUT: usize,
  Obj: Objectives<OBJ_CNT, S>,
  Ter: Terminator<S>,
  Sel: Selector<S>,
  Crs: Crossover<CRS_IN, CRS_OUT, S>,
  Mut: Mutator<S>,
>
{
  /// Runs `Solver` until the termination condition is met. Returns
  /// `SolverResults` from which all or only nondominated solutions can be
  /// extracted.
  fn run(self) -> SolverResults<S>;
}

/// Contains solutions found by `Solver` from which all or only nondominated
/// solutions can be extracted.
pub struct SolverResults<S> {
  solutions: Vec<S>,
  last_nondom_idx: usize,
}

impl<S> SolverResults<S> {
  pub fn all_solutions(self) -> Vec<S> {
    self.solutions
  }

  pub fn nondominated_solutions(self) -> Vec<S> {
    let mut this = self;
    this.solutions.truncate(this.last_nondom_idx);
    this.solutions
  }
}
