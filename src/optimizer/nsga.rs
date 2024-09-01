use std::{cmp::Ordering, collections::HashSet, marker::PhantomData};

use typed_builder::TypedBuilder;

use super::Optimizer;
use crate::{
  mutator::MutationExecutor,
  recombinator::RecombinationExecutor,
  score::{ParetoDominance, Scores},
  selector::SelectionExecutor,
  terminator::TerminationExecutor,
  tester::TestExecutor,
};

#[derive(TypedBuilder)]
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
  population: Vec<Solution>,
  tester: Tst,
  selector: Sel,
  recombinator: Rec,
  mutator: Mut,
  terminator: Ter,
  #[builder(setter(skip), default = vec![])]
  scores: Vec<Scores<OBJECTIVE_NUM>>,
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
  fn peek_population(&self) -> &[Solution] {
    &self.population
  }

  fn peek_scores(&self) -> &[Scores<OBJECTIVE_NUM>] {
    &self.scores
  }

  fn take_population(&mut self) -> Vec<Solution> {
    std::mem::take(&mut self.population)
  }

  fn take_scores(&mut self) -> Vec<Scores<OBJECTIVE_NUM>> {
    std::mem::take(&mut self.scores)
  }

  fn set_population(&mut self, population: Vec<Solution>) {
    self.population = population;
  }

  fn set_scores(&mut self, scores: Vec<Scores<OBJECTIVE_NUM>>) {
    self.scores = scores;
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

  fn create(&self, population: &[&Solution]) -> Vec<Solution> {
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

    let mut some_solutions: Vec<_> = solutions.into_iter().map(Some).collect();
    let mut some_scores: Vec<_> = scores.into_iter().map(Some).collect();
    let (new_sols, new_scs): (Vec<Solution>, Vec<_>) = new_solutions_indices
      .into_iter()
      .map(|idx| {
        (
          some_solutions[idx].take().expect("must be something here"),
          some_scores[idx].take().expect("must be something here"),
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
