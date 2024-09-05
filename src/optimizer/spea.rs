//! Implementations of genetic algorithms of SPEA family.

use std::{marker::PhantomData, ops::Not};

use typed_builder::TypedBuilder;

use crate::{
  mutation::executor::MutationExecutor,
  optimizer::genetic_algorithm::GeneticOptimizer,
  recombination::executor::RecombinationExecutor,
  score::Scores,
  selection::executor::SelectionExecutor,
  termination::executor::TerminationExecutor,
  testing::executor::TestExecutor,
};

/// TODO: add docs
#[derive(TypedBuilder, Debug)]
pub struct Spea2<
  Solution,
  Tst: TestExecutor<Solution, OBJECTIVE_NUM, TstExecStrat>,
  Sel: SelectionExecutor<Solution, OBJECTIVE_NUM, SelExecStrat>,
  Rec: RecombinationExecutor<Solution, PARENT_NUM, OFFSPRING_NUM, RecExecStrat>,
  Mut: MutationExecutor<Solution, MutExecStrat>,
  Ter: TerminationExecutor<Solution, OBJECTIVE_NUM, TerExecStrat>,
  TstExecStrat,
  TerExecStrat,
  SelExecStrat,
  MutExecStrat,
  RecExecStrat,
  const OBJECTIVE_NUM: usize,
  const PARENT_NUM: usize,
  const OFFSPRING_NUM: usize,
> {
  #[builder(setter(
    transform = |v: Vec<Solution>| {
      v.is_empty()
        .not()
        .then_some(v)
        .unwrap_or_else(|| panic!("initial population is empty"))
    },
    doc = "
The initial population setter.
  
# Panics

Panics if population is empty.",    
  ))]
  population: Vec<Solution>,
  tester: Tst,
  selector: Sel,
  recombinator: Rec,
  mutator: Mut,
  terminator: Ter,
  #[builder(setter(skip), default = vec![])]
  scores: Vec<Scores<OBJECTIVE_NUM>>,
  #[builder(setter(skip), default)]
  _solution: PhantomData<Solution>,
  #[builder(setter(skip), default)]
  _eva_es: PhantomData<TstExecStrat>,
  #[builder(setter(skip), default)]
  _ter_es: PhantomData<TerExecStrat>,
  #[builder(setter(skip), default)]
  _sel_es: PhantomData<SelExecStrat>,
  #[builder(setter(skip), default)]
  _mut_es: PhantomData<MutExecStrat>,
  #[builder(setter(skip), default)]
  _rec_es: PhantomData<RecExecStrat>,
}

/// Index of solution in `solutions` vector.
type SolutionIndex = usize;
/// Number of dominated solutions.
type StrengthValue = usize;
/// Sum of strength values of dominating solutions and density.
type Fitness = f64;
/// Density of a solution.
type Density = f64;

impl<
    Solution,
    Tst: TestExecutor<Solution, OBJECTIVE_NUM, TstExecStrat>,
    Sel: SelectionExecutor<Solution, OBJECTIVE_NUM, SelExecStrat>,
    Rec: RecombinationExecutor<Solution, PARENT_NUM, OFFSPRING_NUM, RecExecStrat>,
    Mut: MutationExecutor<Solution, MutExecStrat>,
    Ter: TerminationExecutor<Solution, OBJECTIVE_NUM, TerExecStrat>,
    TstExecStrat,
    TerExecStrat,
    SelExecStrat,
    MutExecStrat,
    RecExecStrat,
    const OBJECTIVE_NUM: usize,
    const PARENT_NUM: usize,
    const OFFSPRING_NUM: usize,
  > GeneticOptimizer<Solution, OBJECTIVE_NUM>
  for Spea2<
    Solution,
    Tst,
    Sel,
    Rec,
    Mut,
    Ter,
    TstExecStrat,
    TerExecStrat,
    SelExecStrat,
    MutExecStrat,
    RecExecStrat,
    OBJECTIVE_NUM,
    PARENT_NUM,
    OFFSPRING_NUM,
  >
{
  fn take_initial_population(&mut self) -> Vec<Solution> {
    std::mem::take(&mut self.population)
  }

  fn test(&self, population: &[Solution]) -> Vec<Scores<OBJECTIVE_NUM>> {
    self.tester.execute_tests(population)
  }

  fn select<'a>(
    &mut self,
    population: &'a [Solution],
    scores: &[Scores<OBJECTIVE_NUM>],
  ) -> Vec<&'a Solution> {
    self.selector.execute_selection(population, scores)
  }

  fn create(&self, population: Vec<&Solution>) -> Vec<Solution> {
    self.recombinator.execute_recombination(population)
  }

  fn mutate(&self, population: &mut [Solution]) {
    self.mutator.execute_mutations(population)
  }

  fn terminate(&mut self) -> bool {
    self
      .terminator
      .execute_termination(&self.population, &self.scores)
  }

  fn truncate(
    &self,
    solutions: Vec<Solution>,
    scores: Vec<Scores<OBJECTIVE_NUM>>,
  ) -> (Vec<Solution>, Vec<Scores<OBJECTIVE_NUM>>) {
    todo!()
  }
}
