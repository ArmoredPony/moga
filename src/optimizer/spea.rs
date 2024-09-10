//! Implementations of genetic algorithms of SPEA family.

use std::{cmp::Ordering, marker::PhantomData, ops::Not};

use typed_builder::TypedBuilder;

use super::Optimizer;
use crate::{
  mutation::executor::MutationExecutor,
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
  #[builder(setter(
    transform = |v: usize| {
      if v == 0 {
        panic!("archive size cannot be 0")
      }
      v
    },
    doc = "
      The archive size setter.

      # Panics

      Panics if archive size is 0.",    
  ))]
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
  >
  Spea2<
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
  fn environmental_selection<'a>(
    &self,
    solutions: &'a [Solution],
    scores: &[Scores<OBJECTIVE_NUM>],
  ) -> (Vec<&'a Solution>) {
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
    let mut sol_indices_fitness: Vec<(SolutionIndex, Fitness)> = solutions
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
            sol_indices_fitness[q_idx].1 += f64::from(strength_values[p_idx])
          }
          Ordering::Greater => {
            sol_indices_fitness[p_idx].1 += f64::from(strength_values[q_idx])
          }
          Ordering::Equal => {}
        }
      }
    }

    // sort and truncate solution by their fitness
    sol_indices_fitness
      .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).expect("NaN encountered"));
    let nondommed_idx = sol_indices_fitness.partition_point(|(_, f)| *f < 1.0);
    // is there are more nondommed solutions than the archive can fit...
    if nondommed_idx > self.archive_size {
      // TODO: densities
    }
    // otherwise truncate solutions so the least dominated ones remain
    // if there are less solutions than the archive size, then this operation
    // won't truncate anything, and the archive will not be full
    else {
      sol_indices_fitness.truncate(self.archive_size);
    }

    debug_assert!(
      sol_indices_fitness.len() <= self.archive_size,
      "new archive population size cannot be bigger than the archive size"
    );

    sol_indices_fitness
      .into_iter()
      .map(|(idx, _)| &solutions[idx])
      .collect()
  }
}

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
  > Optimizer<Solution, OBJECTIVE_NUM>
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
  /// Runs SPEA-II `Optimizer` until the termination condition is met, then
  /// returns the last found population.
  ///
  /// # Panics
  ///
  /// Panic if at some point the population becomes empty, or the number of
  /// scores doesn't match the population size.
  fn optimize(mut self) -> Vec<Solution> {
    let mut population = std::mem::take(&mut self.population);
    let mut scores = self.tester.execute_tests(&population);

    while !self.terminator.execute_termination(&population, &scores) {
      assert!(!population.is_empty(), "the population is empty");
      assert_eq!(
        scores.len(),
        population.len(),
        "the number of calculated fitness scores doesn't match size of the population"
      );
      let survived_solutions =
        self.environmental_selection(&population, &scores);
      let selected_population =
        self.selector.execute_selection(&population, &scores);
      let mut created_population =
        self.recombinator.execute_recombination(selected_population);
      self.mutator.execute_mutations(&mut created_population);
      let mut created_scores = self.tester.execute_tests(&created_population);

      population.append(&mut created_population);
      scores.append(&mut created_scores);

      // (population, scores) =
      //   self.crowding_distance_selection(population, scores);
    }

    population
  }
}
