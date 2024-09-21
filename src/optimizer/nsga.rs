//! Implementations of genetic algorithms of NSGA family.

use std::{cmp::Ordering, collections::HashSet, marker::PhantomData, ops::Not};

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

/// An implementation of a fast and elitist multiobjective genetic algorithm -
/// [NSGA-II].
///
/// [NSGA-II]: https://cs.uwlax.edu/~dmathias/cs419/readings/NSGAIIElitistMultiobjectiveGA.pdf
///
/// # Examples
///
/// *Schaffer's Problem No.1* solution.
/// ```no_run
/// # fn main() {
/// use rand::Rng;
/// use moga::{
///   operator::ParBatch,
///   optimizer::{nsga::Nsga2, Optimizer},
///   selection::RandomSelector,
///   termination::GenerationTerminator,
/// };
/// // initial solutions lie between 0 and 100
/// let population = (0..100).map(|i| i as f32).collect::<Vec<_>>();
/// // objective functions `f1(x) = x^2` and `f2(x) = (x - 2)^2`
/// let test = |x: &f32| [x.powf(2.0), (x - 2.0).powf(2.0)];
/// // a `Selector` that selects 10 random solutions
/// let selector = RandomSelector(10);
/// // for each pair of parents `x` and `y` create an offspring `o = x + r * (y - x)`
/// // where `r` is a random value between -1 and 2
/// let r = || rand::thread_rng().gen_range(-1.0..2.0);
/// let recombinator = |x: &f32, y: &f32| x + r() * (y - x);
/// // a `Mutation` that does not mutate solutions
/// let mutation = |_: &mut f32| {};
/// // a `Termiantor` that terminates after 100 generations
/// let terminator = GenerationTerminator(100);
/// // a convinient builder with compile time verification from `typed-builder` crate
/// let optimizer = Nsga2::builder()
///   .population(population)
///   // `test` will be executed concurrently for each batch of solutions
///   .tester(test.par_batch())
///   .selector(selector)
///   .recombinator(recombinator)
///   .mutator(mutation)
///   .terminator(terminator)
///   .build();
/// // upon termination the optimizer returns the best solutions it has found
/// let solutions = optimizer.optimize();
/// # }
/// ```
#[derive(TypedBuilder, Debug)]
pub struct Nsga2<
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
  #[builder(setter(skip), default = population.len())]
  initial_population_size: usize,
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
/// Number of solution's dominators.
type DominanceCounter = u32;
/// Crowding distance of a solution.
type CrowdingDistance = f64;
/// Front number. the lower - the better.
type FrontNumber = u32;
/// Dominated by each solution solutions' indices.
type DominanceList = Vec<SolutionIndex>;
/// Indices of solutions of a front.
type Front = Vec<SolutionIndex>;

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
  Nsga2<
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
  fn crowding_distance_selection(
    &self,
    solutions: Vec<Solution>,
    scores: Vec<Scores<OBJECTIVE_NUM>>,
  ) -> (Vec<Solution>, Vec<Scores<OBJECTIVE_NUM>>) {
    let mut dominance_lists: Vec<DominanceList> =
      vec![Vec::new(); solutions.len()];
    let mut dominance_counters: Vec<DominanceCounter> =
      vec![0; solutions.len()];
    let mut first_front: Front = Vec::new();

    // fill dominance sets and counters
    for p_idx in 0..solutions.len() {
      // for each unique pair of solutions `p`...
      let (p_sc, rest_scs) =
        scores[p_idx..].split_first().expect("no scores remain");
      // and `q`...
      for (i, q_sc) in rest_scs.iter().enumerate() {
        let q_idx = p_idx + i + 1;
        match p_sc.dominance(q_sc) {
          // if solution `p` dominates solution `q`...
          Ordering::Less => {
            // put solution `q` into list of solutions dominated by `p`
            dominance_lists[p_idx].push(q_idx);
            // and increment counter of solutions dominating `q`
            dominance_counters[q_idx] += 1;
          }
          // if solution `q` dominates solution `p`...
          Ordering::Greater => {
            // put solution `p` into list of solutions dominated by `q`
            dominance_lists[q_idx].push(p_idx);
            // and increment counter of solutions dominating `p`
            dominance_counters[p_idx] += 1;
          }
          Ordering::Equal => {}
        }
      }
      // if solution `p` isn't dominated by any other solution...
      if dominance_counters[p_idx] == 0 {
        first_front.push(p_idx); // put its index into the first front
      }
    }

    debug_assert!(
      !first_front.is_empty(),
      "first front must have at least 1 solution"
    );

    let mut front_numbers: Vec<FrontNumber> =
      vec![FrontNumber::MAX; solutions.len()];
    let mut last_front = first_front;
    let mut new_solutions_indices: Vec<SolutionIndex> = Vec::new();
    let mut front_idx = 0;
    // until we select enough solutions...
    while new_solutions_indices.len() + last_front.len()
      < self.initial_population_size
    {
      let mut next_front = Vec::new();
      // for each solution `p` in last front...
      for p_idx in last_front.iter() {
        // for each solution `q` dominated by `p`...
        for q_idx in dominance_lists[*p_idx].iter_mut() {
          // decrement counter of solutions dominating `q`
          dominance_counters[*q_idx] -= 1;
          // if no more solutions dominate `q`...
          if dominance_counters[*q_idx] == 0 {
            front_numbers[*q_idx] = front_idx; // set front number of solution
            next_front.push(*q_idx); // and push its index into next front
          }
        }
      }
      // save current last front indices
      new_solutions_indices.append(&mut last_front);
      // and replace last front with next front
      last_front = next_front;
      front_idx += 1;
    }

    // calculate crowding distance for each solution in the last found front
    let mut crowding_distances: Vec<CrowdingDistance> =
      vec![0.0; solutions.len()];
    // if last front has more than 2 values...
    if last_front.len() > 2 {
      // for each objective `o`...
      for o_idx in 0..OBJECTIVE_NUM {
        // sort solutions by their scores of objective `o`
        last_front.sort_by(|&a_idx, &b_idx| {
          scores[a_idx][o_idx]
            .partial_cmp(&scores[b_idx][o_idx])
            .unwrap_or(Ordering::Greater) // sort NaNs away
        });

        // get the first and the last front members
        let first_idx = last_front[0];
        let last_idx = last_front[last_front.len() - 1];
        // set crowding distances of first and last front member to `f64::MAX`
        crowding_distances[first_idx] = f64::MAX;
        crowding_distances[last_idx] = f64::MAX;
        // calculate difference between max and min score of current objective
        let min_score = scores[first_idx][o_idx];
        let max_score = scores[last_idx][o_idx];
        let score_diff = if max_score != min_score {
          f64::from(max_score - min_score)
        } else {
          1.0
        };
        // for each solution except the first and the last in the last front...
        for idx in 2..last_front.len() - 2 {
          if crowding_distances[idx] != f64::MAX {
            let prev_cd = crowding_distances[idx - 1];
            let next_cd = crowding_distances[idx + 1];
            crowding_distances[idx] += (next_cd - prev_cd).abs() / score_diff;
          }
        }
      }
      // sort solutions in the last front by their crowding distances
      last_front.sort_by(|&a_idx, &b_idx| {
        crowding_distances[b_idx].total_cmp(&crowding_distances[a_idx])
      });
    }

    new_solutions_indices.append(&mut last_front);
    new_solutions_indices.truncate(self.initial_population_size);
    new_solutions_indices.sort_by(|&a_idx, &b_idx| {
      front_numbers[a_idx]
        .cmp(&front_numbers[b_idx])
        .then(crowding_distances[b_idx].total_cmp(&crowding_distances[a_idx]))
    });

    debug_assert_eq!(
      new_solutions_indices.len(),
      HashSet::<usize>::from_iter(new_solutions_indices.iter().cloned()).len(),
      "new_solutions_indices must have only unique indices"
    );

    let mut some_sols: Vec<_> = solutions.into_iter().map(Some).collect();
    let mut some_scs: Vec<_> = scores.into_iter().map(Some).collect();
    let (new_sols, new_scs): (Vec<_>, Vec<_>) = new_solutions_indices
      .into_iter()
      .map(|idx| {
        (
          some_sols[idx].take().expect("must be something here"),
          some_scs[idx].take().expect("must be something here"),
        )
      })
      .unzip();

    debug_assert_eq!(
      new_sols.len(),
      self.initial_population_size,
      "new population size must match initial population size"
    );
    debug_assert_eq!(
      new_sols.len(),
      new_scs.len(),
      "number of solutions must match number of scores"
    );

    (new_sols, new_scs)
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
  for Nsga2<
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
  /// Runs NSGA-II `Optimizer` until the termination condition is met, then
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
      let selected_population =
        self.selector.execute_selection(&population, &scores);
      let mut created_population =
        self.recombinator.execute_recombination(selected_population);
      self.mutator.execute_mutations(&mut created_population);
      let mut created_scores = self.tester.execute_tests(&created_population);

      population.append(&mut created_population);
      scores.append(&mut created_scores);

      (population, scores) =
        self.crowding_distance_selection(population, scores);
    }

    population
  }
}
