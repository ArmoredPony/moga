mod nsga;

use std::cmp::Ordering;

use crate::{
  crossover::Crossover, mutator::Mutator, objective::Objective,
  selector::Selector, terminator::Terminator,
};

/// Represents an abstract problem solver.
pub trait Solver<
  S,
  const OBJ_CNT: usize,
  const CRS_IN: usize,
  const CRS_OUT: usize,
  Obj: Objective<OBJ_CNT, S>,
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

trait ParetoDominance<S> {
  fn dominance(&self, other: &Self) -> Ordering;
}

impl<const N: usize, S: PartialOrd> ParetoDominance<S> for [S; N] {
  fn dominance(&self, other: &Self) -> Ordering {
    self
      .iter()
      .zip(other)
      .find_map(|(a, b)| {
        a.partial_cmp(b)
          .map(|ord| match ord {
            Ordering::Equal => None,
            ord => Some(ord),
          })
          .unwrap_or(Some(Ordering::Equal))
      })
      .unwrap_or(Ordering::Equal)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_pareto_dominance() {
    assert_eq!([1.0, 2.0, 3.0].dominance(&[1.0, 2.0, 3.0]), Ordering::Equal);
    assert_eq!(
      [10.0, 2.0, 3.0].dominance(&[1.0, 2.0, 3.0]),
      Ordering::Greater
    );
    assert_eq!(
      [1.0, 20.0, 3.0].dominance(&[1.0, 2.0, 3.0]),
      Ordering::Greater
    );
    assert_eq!(
      [1.0, 2.0, 30.0].dominance(&[1.0, 2.0, 3.0]),
      Ordering::Greater
    );
    assert_eq!([1.0, 2.0, 3.0].dominance(&[10.0, 2.0, 3.0]), Ordering::Less);
    assert_eq!([1.0, 2.0, 3.0].dominance(&[1.0, 20.0, 3.0]), Ordering::Less);
    assert_eq!([1.0, 2.0, 3.0].dominance(&[1.0, 2.0, 30.0]), Ordering::Less);
    assert_eq!([1i32; 0].dominance(&[0i32; 0]), Ordering::Equal);
    assert_eq!(
      [f32::NAN, 20.0, 30.0].dominance(&[1.0, 2.0, 3.0]),
      Ordering::Equal
    );
  }
}
