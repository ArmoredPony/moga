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
/// Distance from k-th solution.
type Distance = f64;

impl<
    Solution: Clone,
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
  fn environmental_selection(
    &self,
    solutions: &[Solution],
    scores: &[Scores<OBJECTIVE_NUM>],
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
    let mut sol_indices_fitness: Vec<(SolutionIndex, Fitness)> =
      (0..solutions.len()).map(|i| (i, 0.0)).collect();
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
    let nondommed_cnt = sol_indices_fitness.partition_point(|(_, f)| *f == 0.0);
    let mut new_sol_indicies: Vec<_> = sol_indices_fitness
      .into_iter()
      .map(|(idx, _)| idx)
      .collect();
    // is there are more nondommed solutions than the archive can fit...
    if nondommed_cnt >= self.archive_size {
      let k = ((solutions.len() + self.archive_size) as f32).sqrt() as usize;
      let dommed_sol = new_sol_indicies.split_off(nondommed_cnt);

      let mut distances: Vec<(SolutionIndex, Vec<Distance>)> = (0
        ..new_sol_indicies.len())
        .map(|i| (i, Vec::with_capacity(new_sol_indicies.len() - 1)))
        .collect();

      for i in 0..new_sol_indicies.len() {
        let (p_idx, rest) = new_sol_indicies[i..]
          .split_first()
          .expect("no solutions remain");

        for (j, q_idx) in rest.iter().enumerate() {
          let j = i + j + 1;
          let p_sc = scores[*p_idx];
          let q_sc = scores[*q_idx];
          let d = p_sc
            .iter()
            .zip(q_sc)
            .map(|(a, b)| (a - b).powf(2.0) as f64)
            .sum();
          distances[i].1.push(d);
          distances[j].1.push(d);
        }
        for d_idx in &dommed_sol {
          if distances[i].1.len() >= k {
            break;
          }
          let p_sc = scores[*p_idx];
          let d_sc = scores[*d_idx];
          let d = p_sc
            .iter()
            .zip(d_sc)
            .map(|(a, b)| (a - b).powf(2.0) as f64)
            .sum();
          distances[i].1.push(d);
        }
        distances[i]
          .1
          .sort_unstable_by(|a, b| a.partial_cmp(b).expect("NaN encountered"));
      }
      distances
        .sort_by(|a, b| b.1[k].partial_cmp(&a.1[k]).expect("NaN encountered"));
      new_sol_indicies = distances
        .into_iter()
        .take(self.archive_size)
        .map(|(idx, _)| idx)
        .collect();

      // while new_sol_indicies.len() > self.archive_size {
      //   let mut distances: Vec<(SolutionIndex, Vec<f32>)> = (0
      //     ..new_sol_indicies.len())
      //     .map(|i| (i, Vec::with_capacity(new_sol_indicies.len() - 1)))
      //     .collect();

      //   for i in 0..new_sol_indicies.len() {
      //     let (p_idx, rest) = new_sol_indicies[i..]
      //       .split_first()
      //       .expect("no solutions remain");

      //     for (j, q_idx) in rest.iter().enumerate() {
      //       let j = i + j + 1;
      //       let p_sc = scores[*p_idx];
      //       let q_sc = scores[*q_idx];
      //       let d: f32 =
      //         p_sc.iter().zip(q_sc).map(|(a, b)| (a - b).powf(2.0)).sum();
      //       distances[i].1.push(d);
      //       distances[j].1.push(d);
      //     }
      //     for d_idx in &dommed_sol {
      //       if distances[i].1.len() >= k {
      //         break;
      //       }
      //       let p_sc = scores[*p_idx];
      //       let d_sc = scores[*d_idx];
      //       let d: f32 =
      //         p_sc.iter().zip(d_sc).map(|(a, b)| (a - b).powf(2.0)).sum();
      //       distances[i].1.push(d);
      //     }
      //     distances[i].1.sort_unstable_by(|a, b| {
      //       a.partial_cmp(b).expect("NaN encountered")
      //     });
      //     // distances[i].1.truncate(k);
      //   }

      //   let removed_idx = distances
      //     .iter()
      //     .min_by(|a, b| {
      //       for (i, j) in a.1[..k].iter().rev().zip(b.1[..k].iter().rev()) {
      //         match i.partial_cmp(j).expect("NaN encountered") {
      //           Ordering::Less => return Ordering::Less,
      //           Ordering::Greater => return Ordering::Greater,
      //           Ordering::Equal => {}
      //         }
      //       }
      //       Ordering::Less
      //     })
      //     .expect("no solutions remain")
      //     .0;
      //   new_sol_indicies.remove(removed_idx);
      // }

      debug_assert_eq!(new_sol_indicies.len(), self.archive_size);
    }
    // otherwise truncate solutions so the least dominated ones remain
    // if there are less solutions than the archive size, then this operation
    // won't truncate anything, and the archive will not be full
    else {
      new_sol_indicies.truncate(self.archive_size);
    }

    debug_assert!(
      new_sol_indicies.len() <= self.archive_size,
      "new archive population size cannot be bigger than the archive size"
    );

    new_sol_indicies
      .into_iter()
      // TODO: reimplement operators and remove cloning
      .map(|idx| (solutions[idx].clone(), scores[idx]))
      .unzip()
  }
}

impl<
    Solution: Clone,
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
    let mut population_scores = self.tester.execute_tests(&population);

    let mut archive: Vec<Solution> = Vec::new();
    let mut archive_scores: Vec<Scores<OBJECTIVE_NUM>> = Vec::new();

    while !self
      .terminator
      .execute_termination(&archive, &archive_scores)
    {
      archive.append(&mut population);
      archive_scores.append(&mut population_scores);

      assert!(!archive.is_empty(), "the population is empty");
      assert_eq!(
        archive_scores.len(),
        archive.len(),
        "the number of calculated fitness scores doesn't match size of the population"
      );
      let (survived_solutions, survived_scores) =
        self.environmental_selection(&archive, &archive_scores);
      let selected_solutions = self
        .selector
        .execute_selection(&survived_solutions, &survived_scores);
      let mut created_solutions =
        self.recombinator.execute_recombination(selected_solutions);
      self.mutator.execute_mutations(&mut created_solutions);
      let created_scores = self.tester.execute_tests(&created_solutions);

      archive = survived_solutions;
      archive_scores = survived_scores;
      population = created_solutions;
      population_scores = created_scores;
    }

    archive
  }
}
