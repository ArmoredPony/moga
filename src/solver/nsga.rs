use std::marker::PhantomData;

use super::Solver;

use crate::{
  crossover::Crossover, mutator::Mutator, objective::Objective,
  selector::Selector, terminator::Terminator,
};

pub struct Nsga2<
  S,
  const OBJ_CNT: usize,
  const CRS_IN: usize,
  const CRS_OUT: usize,
  Obj: Objective<OBJ_CNT, S>,
  Ter: Terminator<S>,
  Sel: Selector<S>,
  Crs: Crossover<CRS_IN, CRS_OUT, S>,
  Mut: Mutator<S>,
> {
  objective: Obj,
  terminator: Ter,
  selector: Sel,
  crossover: Crs,
  mutator: Mut,
  _solution: PhantomData<S>,
}

impl<
    S,
    const OBJ_CNT: usize,
    const CRS_IN: usize,
    const CRS_OUT: usize,
    Obj: Objective<OBJ_CNT, S>,
    Ter: Terminator<S>,
    Sel: Selector<S>,
    Crs: Crossover<CRS_IN, CRS_OUT, S>,
    Mut: Mutator<S>,
  > Solver<S, OBJ_CNT, CRS_IN, CRS_OUT, Obj, Ter, Sel, Crs, Mut>
  for Nsga2<S, OBJ_CNT, CRS_IN, CRS_OUT, Obj, Ter, Sel, Crs, Mut>
{
  fn run(self) -> super::SolverResults<S> {
    todo!()
  }
}
