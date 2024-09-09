//! Implementations of genetic algorithms of SPEA family.

use std::{cmp::Ordering, marker::PhantomData, ops::Not};

use typed_builder::TypedBuilder;

use crate::{
  mutation::executor::MutationExecutor,
  optimizer::genetic_algorithm::GeneticOptimizer,
  recombination::executor::RecombinationExecutor,
  score::{ParetoDominance, Scores},
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
  archive_size: usize,
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
type StrengthValue = u32;
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
    // each i-th value is a number of solutions that i-th solution dominates
    let mut strength_values: Vec<StrengthValue> = vec![0; solutions.len()];
    // count strength values for each solution
    for p_idx in 0..solutions.len() - 1 {
      let (p_sc, rest_scs) =
        scores[p_idx..].split_first().expect("no scores remain");
      for (i, q_sc) in rest_scs.iter().enumerate() {
        let q_idx = p_idx + i + 1;
        match p_sc.dominance(q_sc) {
          Ordering::Less => strength_values[p_idx] += 1,
          Ordering::Greater => strength_values[q_idx] += 1,
          Ordering::Equal => {}
        }
      }
    }

    // each i-th value is a sum of strength values of solutions that dominate
    // i-th solution
    let mut solution_fitness: Vec<(SolutionIndex, Fitness)> = solutions
      .iter()
      .enumerate()
      .map(|(i, _)| (i, 0.0))
      .collect();
    // compute raw fitness for each solution
    for p_idx in 0..solutions.len() - 1 {
      let (p_sc, rest_scs) =
        scores[p_idx..].split_first().expect("no scores remain");
      for (i, q_sc) in rest_scs.iter().enumerate() {
        let q_idx = p_idx + i + 1;
        match p_sc.dominance(q_sc) {
          Ordering::Less => {
            solution_fitness[q_idx].1 += f64::from(strength_values[p_idx])
          }
          Ordering::Greater => {
            solution_fitness[p_idx].1 += f64::from(strength_values[q_idx])
          }
          Ordering::Equal => {}
        }
      }
    }

    // truncate solution by their fitness
    solution_fitness
      .sort_by(|a, b| a.1.partial_cmp(&b.1).expect("NaN encountered"));
    let nondommed_idx = solution_fitness.partition_point(|(_, f)| *f == 0.0);
    // is there are more nondommed solutions than the archive can fit...
    if nondommed_idx > self.archive_size {
      // TODO: densities
    }

    todo!()
  }
}
