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

/// An implementation of an improved version of the Strength Pareto Evolutionary
/// Algorithm - [SPEA-II].
///
/// [SPEA-II]: https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/145755/eth-24689-01.pdf
///
/// # Examples
///
/// *Schaffer's Problem No.1* solution.
/// ```no_run
/// # use rand::{seq::IteratorRandom, Rng};
/// # use moga::{
/// #  optimizer::spea::Spea2,
/// #  selection::RouletteSelector,
/// #  termination::GenerationTerminator,
/// #  Optimizer,
/// #  ParBatch,
/// # };
/// # fn main() {
/// // initial solutions lie between 0 and 100
/// let population = (0..100).map(|i| i as f32).collect::<Vec<_>>();
/// // archive size of `Spea2` optimizer
/// let archive_size = 100;
/// // objective functions `f1(x) = x^2` and `f2(x) = (x - 2)^2`
/// let test = |x: &f32| [x.powf(2.0), (x - 2.0).powf(2.0)];
/// // a `Selector` that selects 10 random solutions. selection chance of a
/// // solution is directly proportional to the number of solutions it dominates
/// let selector = RouletteSelector(10);
/// // for each pair of parents `x` and `y` create an offspring
/// // `o = x + r * (y - x)` where `r` is a random value between -1 and 2
/// let r = || rand::thread_rng().gen_range(-1.0..2.0);
/// let recombinator = |x: &f32, y: &f32| x + r() * (y - x);
/// // a `Mutation` that does not mutate solutions
/// let mutation = |_: &mut f32| {};
/// // a `Termiantor` that terminates after 100 generations
/// let terminator = GenerationTerminator(100);
/// // a convinient builder with compile time verification from `typed-builder` crate
/// let spea2 = Spea2::builder()
/// .population(population)
/// .archive_size(archive_size)
/// // `test` will be executed concurrently for each batch of solutions
/// .tester(test.par_batch())
/// .selector(selector)
/// .recombinator(recombinator)
/// .mutator(mutation)
/// .terminator(terminator)
/// .build();
/// // upon termination optimizer returns the best solutions it has found
/// let solutions = spea2.optimize();
/// # }
/// ```
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
  /// Environmental selection procedure of SPEA-II algorithm.
  fn environmental_selection(
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
    // i-th solution paired with index of a solution in `solutions`
    let mut sol_idx_fit: Vec<(SolutionIndex, Fitness)> =
      (0..solutions.len()).map(|i| (i, 0.0)).collect();
    // compute raw fitness for each solution
    for p_idx in 0..solutions.len() - 1 {
      let (p_sc, rest_scs) =
        scores[p_idx..].split_first().expect("no scores remain");
      for (i, q_sc) in rest_scs.iter().enumerate() {
        let q_idx = p_idx + i + 1;
        match p_sc.dominance(q_sc) {
          Ordering::Less => {
            sol_idx_fit[q_idx].1 += f64::from(strength_values[p_idx])
          }
          Ordering::Greater => {
            sol_idx_fit[p_idx].1 += f64::from(strength_values[q_idx])
          }
          Ordering::Equal => {}
        }
      }
    }

    // count nondominated solutions
    let nondommed_cnt = sol_idx_fit.iter().filter(|(_, f)| *f < 1.0).count();
    let new_sol_idx_fit: Vec<(SolutionIndex, Distance)> =
      if nondommed_cnt > self.archive_size {
        // if there are more nondommed solutions than the archive can fit, truncate
        // solutions iteratively by their distance to k-th neighbor
        let mut nondom_idx_fit: Vec<_> =
          sol_idx_fit.into_iter().filter(|(_, f)| *f < 1.0).collect();

        // while there are more solutions than the archive size...
        while nondom_idx_fit.len() > self.archive_size {
          // get vector of distances for each solution
          let sol_distances = sorted_sol_distances(&nondom_idx_fit, &scores);
          // find index of a solution with smallest distance to another solution
          let removed_idx = sol_distances
            .iter()
            .min_by(|a, b| {
              for (i, j) in a.1.iter().zip(b.1.iter()) {
                match i.partial_cmp(j).expect("NaN encountered") {
                  Ordering::Less => return Ordering::Less,
                  Ordering::Greater => return Ordering::Greater,
                  Ordering::Equal => {}
                }
              }
              Ordering::Equal
            })
            .expect("no solutions remain")
            .0;
          // remove solution by this index
          nondom_idx_fit.swap_remove(removed_idx);
        }

        debug_assert_eq!(nondom_idx_fit.len(), self.archive_size);
        nondom_idx_fit
      } else {
        // calculate and add distance to the k-th neighbor to solutions' fitness
        // values
        let sol_distances = sorted_sol_distances(&sol_idx_fit, &scores);
        let k = (sol_idx_fit.len() as f64).sqrt() as usize;
        sol_distances.into_iter().for_each(|(idx, distances)| {
          sol_idx_fit[idx].1 += 1.0 / (distances[k - 1] + 2.0);
        });
        // sort and truncate solutions. if there are less solutions than the
        // archive size, the archive will be partially filled
        sol_idx_fit.sort_unstable_by(|a, b| {
          a.1.partial_cmp(&b.1).expect("NaN encountered")
        });
        sol_idx_fit.truncate(self.archive_size);
        sol_idx_fit
      };

    debug_assert!(
      new_sol_idx_fit.len() <= self.archive_size,
      "new archive population size cannot be bigger than the archive size"
    );

    let mut solutions = solutions.into_iter().map(Some).collect::<Vec<_>>();
    new_sol_idx_fit
      .into_iter()
      .map(|(idx, _)| {
        (
          std::mem::take(&mut solutions[idx]).expect("must be something here"),
          scores[idx],
        )
      })
      .unzip()
  }
}

/// Calculates and returns sorted distances between solutions.
/// Each element of returned vector contains index of a solution from
/// `sol_indices` and sorted vector of distances to other solutions.
#[inline]
fn sorted_sol_distances<const N: usize>(
  sol_idx_fit: &[(SolutionIndex, Fitness)],
  scores: &[[f32; N]],
) -> Vec<(SolutionIndex, Vec<Distance>)> {
  let mut sol_distances: Vec<_> = (0..sol_idx_fit.len())
    .map(|i| (i, Vec::with_capacity(sol_idx_fit.len() - 1)))
    .collect();

  for i in 0..sol_idx_fit.len() {
    let (p, rest) =
      sol_idx_fit[i..].split_first().expect("no solutions remain");

    for (j, q) in rest.iter().enumerate() {
      let j = i + j + 1;
      let p_sc = scores[p.0];
      let q_sc = scores[q.0];
      let d: Distance = p_sc
        .iter()
        .zip(q_sc)
        .map(|(a, b)| (a - b).powf(2.0) as f64)
        .sum();
      sol_distances[i].1.push(d);
      sol_distances[j].1.push(d);
    }

    sol_distances[i]
      .1
      .sort_unstable_by(|a, b| a.partial_cmp(b).expect("NaN encountered"));
  }
  sol_distances
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
  /// returns nondominated solutions.
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
        self.environmental_selection(archive, archive_scores);
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

    let mut selected_sols = vec![true; archive.len()];
    for i in 0..selected_sols.len() - 1 {
      if !selected_sols[i] {
        continue;
      }
      for j in (i + 1)..selected_sols.len() {
        match archive_scores[i].dominance(&archive_scores[j]) {
          Ordering::Less => selected_sols[j] = false,
          Ordering::Greater => selected_sols[i] = false,
          Ordering::Equal => {}
        }
      }
    }

    archive
      .into_iter()
      .zip(selected_sols)
      .filter_map(|(sol, is_selected)| is_selected.then_some(sol))
      .collect()
  }
}
