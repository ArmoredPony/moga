use std::{cmp::Ordering, collections::HashSet, marker::PhantomData};

use crate::{
  evaluator::{pareto::ParetoDominance, Scores},
  Crossover,
  Evaluator,
  Mutator,
  Optimizer,
  Selector,
  Terminator,
};

pub struct Nsga2<
  S,
  const OBJECTIVE_CNT: usize,
  const PARENT_CNT: usize,
  const OFFSPRING_CNT: usize,
  Obj: Evaluator<S, OBJECTIVE_CNT>,
  Ter: Terminator<S, OBJECTIVE_CNT>,
  Sel: Selector<S, OBJECTIVE_CNT>,
  Crs: Crossover<S, PARENT_CNT, OFFSPRING_CNT>,
  Mut: Mutator<S>,
> {
  solutions: Vec<S>,
  scores: Vec<Scores<OBJECTIVE_CNT>>,
  initial_population_size: usize,
  objective: Obj,
  terminator: Ter,
  selector: Sel,
  crossover: Crs,
  mutator: Mut,
  _solution: PhantomData<S>,
}

impl<
    S,
    const OBJECTIVE_CNT: usize,
    const PARENT_CNT: usize,
    const OFFSPRING_CNT: usize,
    Obj: Evaluator<S, OBJECTIVE_CNT>,
    Ter: Terminator<S, OBJECTIVE_CNT>,
    Sel: Selector<S, OBJECTIVE_CNT>,
    Crs: Crossover<S, PARENT_CNT, OFFSPRING_CNT>,
    Mut: Mutator<S>,
  > Optimizer<S>
  for Nsga2<
    S,
    OBJECTIVE_CNT,
    PARENT_CNT,
    OFFSPRING_CNT,
    Obj,
    Ter,
    Sel,
    Crs,
    Mut,
  >
{
  fn run(mut self) -> Vec<S> {
    // probably generalize this algorithm in `Optimizer`
    self.scores = self
      .solutions
      .iter()
      .map(|s| self.objective.evaluate(s))
      .collect();

    while !self.terminator.terminate(&self.solutions, &self.scores) {
      let mut solutions = std::mem::take(&mut self.solutions);
      let mut scores = std::mem::take(&mut self.scores);

      let selected_solutions = self.selector.select(&solutions, &scores);
      let mut created_solutions = self.crossover.create(&selected_solutions);
      created_solutions
        .iter_mut()
        .for_each(|s| self.mutator.mutate(s));
      let mut created_scores: Vec<_> = created_solutions
        .iter()
        .map(|s| self.objective.evaluate(s))
        .collect();

      solutions.append(&mut created_solutions);
      scores.append(&mut created_scores);

      (self.solutions, self.scores) =
        self.select_best_solutions(solutions, scores);
    }

    self.solutions
  }
}

// index of solution in `solutions` vector
type SolutionIndex = usize;
// number of solution's dominators
type DominanceCounter = u32;
// crowding distance of a solution
type CrowdingDistance = f64;
// front number. the lower - the better
type FrontNumber = u32;
// dominated by each solution solutions' indices
type DominanceList = Vec<SolutionIndex>;
// indices of solutions of a front
type Front = Vec<SolutionIndex>;

impl<
    S,
    const OBJECTIVE_CNT: usize,
    const PARENT_CNT: usize,
    const OFFSPRING_CNT: usize,
    Obj: Evaluator<S, OBJECTIVE_CNT>,
    Ter: Terminator<S, OBJECTIVE_CNT>,
    Sel: Selector<S, OBJECTIVE_CNT>,
    Crs: Crossover<S, PARENT_CNT, OFFSPRING_CNT>,
    Mut: Mutator<S>,
  >
  Nsga2<S, OBJECTIVE_CNT, PARENT_CNT, OFFSPRING_CNT, Obj, Ter, Sel, Crs, Mut>
{
  pub fn new(
    initial_population: Vec<S>,
    objective: Obj,
    terminator: Ter,
    selector: Sel,
    crossover: Crs,
    mutator: Mut,
  ) -> Self {
    assert!(
      !initial_population.is_empty(),
      "initial population cannot be empty"
    );
    Self {
      initial_population_size: initial_population.len(),
      solutions: initial_population,
      scores: Vec::new(),
      objective,
      terminator,
      selector,
      crossover,
      mutator,
      _solution: PhantomData,
    }
  }

  fn select_best_solutions(
    &mut self,
    solutions: Vec<S>,
    scores: Vec<Scores<OBJECTIVE_CNT>>,
  ) -> (Vec<S>, Vec<Scores<OBJECTIVE_CNT>>) {
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
      for o_idx in 0..OBJECTIVE_CNT {
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
      front_numbers[a_idx].cmp(&front_numbers[b_idx]).then({
        crowding_distances[b_idx].total_cmp(&crowding_distances[a_idx])
      })
    });

    debug_assert_eq!(
      new_solutions_indices.len(),
      HashSet::<usize>::from_iter(new_solutions_indices.iter().cloned()).len(),
      "new_solutions_indices must have only unique indices"
    );

    let mut some_solutions: Vec<_> = solutions.into_iter().map(Some).collect();
    let mut some_scores: Vec<_> = scores.into_iter().map(Some).collect();
    let (new_sols, new_scs): (Vec<S>, Vec<_>) = new_solutions_indices
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
