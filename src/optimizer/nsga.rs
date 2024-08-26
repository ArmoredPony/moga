use std::{cmp::Ordering, marker::PhantomData};

use crate::{
  objective::{pareto::ParetoDominance, Scores},
  Crossover,
  Mutator,
  Objectives,
  Optimizer,
  Selector,
  Terminator,
};

pub struct Nsga2<
  S,
  const OBJ_CNT: usize,
  const CRS_IN: usize,
  const CRS_OUT: usize,
  Obj: Objectives<OBJ_CNT, S>,
  Ter: for<'a> Terminator<'a, S, OBJ_CNT>,
  Sel: Selector<S, OBJ_CNT>,
  Crs: Crossover<CRS_IN, CRS_OUT, S>,
  Mut: Mutator<S>,
> {
  solutions: Vec<S>,
  scores: Vec<Scores<OBJ_CNT>>,
  new_solutions: Vec<S>,
  initial_population_size: usize,
  objective: Obj,
  terminator: Ter,
  selector: Sel,
  crossover: Crs,
  mutator: Mut,
  _solution: PhantomData<S>,
}

impl<
    S: Sync,
    const OBJ_CNT: usize,
    const CRS_IN: usize,
    const CRS_OUT: usize,
    Obj: Objectives<OBJ_CNT, S> + Sync,
    Ter: for<'a> Terminator<'a, S, OBJ_CNT>,
    Sel: Selector<S, OBJ_CNT>,
    Crs: Crossover<CRS_IN, CRS_OUT, S>,
    Mut: Mutator<S>,
  > Optimizer<S>
  for Nsga2<S, OBJ_CNT, CRS_IN, CRS_OUT, Obj, Ter, Sel, Crs, Mut>
{
  fn run(mut self) -> Vec<S> {
    self.solutions = std::mem::take(&mut self.new_solutions);
    self.scores = self
      .solutions
      .iter()
      .map(|s| self.objective.evaluate(s))
      .collect();

    while !self
      .terminator
      .terminate(self.solutions.iter().zip(self.scores.iter()))
    {
      let mut solutions = std::mem::take(&mut self.solutions);
      let mut scores = std::mem::take(&mut self.scores);

      let selected_solutions =
        self.selector.select(solutions.iter().zip(scores.iter()));
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

impl<
    S: Sync,
    const OBJ_CNT: usize,
    const CRS_IN: usize,
    const CRS_OUT: usize,
    Obj: Objectives<OBJ_CNT, S> + Sync,
    Ter: for<'a> Terminator<'a, S, OBJ_CNT>,
    Sel: Selector<S, OBJ_CNT>,
    Crs: Crossover<CRS_IN, CRS_OUT, S>,
    Mut: Mutator<S>,
  > Nsga2<S, OBJ_CNT, CRS_IN, CRS_OUT, Obj, Ter, Sel, Crs, Mut>
{
  // really really really want to refactor this mess
  fn select_best_solutions(
    &mut self,
    solutions: Vec<S>,
    scores: Vec<Scores<OBJ_CNT>>,
  ) -> (Vec<S>, Vec<Scores<OBJ_CNT>>) {
    type SolutionIndex = usize; // index of solution in `solutions` vector
    type DominanceCounter = u32; // number of solution's dominators
    type CrowdingDistance = f64; // crowding distance of a solution
    type IsBestSolution = bool; // if `true`, solution will be selected

    // contains dominated solutions with their indicies by each solution
    let mut dominance_lists: Vec<Vec<SolutionIndex>>;
    // contains number of solutions dominating each solution
    let mut dominance_counters: Vec<DominanceCounter>;
    // contains crowding distance for each solution
    let mut crowding_distances: Vec<CrowdingDistance>;
    // if a solution belongs to the best ones, the corresponding value is `true`
    let mut best_solutions_list: Vec<IsBestSolution>;
    // indicies of solutions of the first front
    let mut first_front: Vec<SolutionIndex>;

    dominance_lists = vec![Vec::new(); solutions.len()];
    dominance_counters = vec![0; solutions.len()];
    crowding_distances = vec![0.0; solutions.len()];
    best_solutions_list = vec![false; solutions.len()];
    first_front = Vec::new();

    // fill dominance sets and counters
    for p_idx in 0..solutions.len() - 1 {
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

    let mut last_front = first_front;
    let mut front_solutions_count = last_front.len();
    // while last front isn't empty and we haven't found enough solutions...
    while !last_front.is_empty() && front_solutions_count < solutions.len() {
      let mut next_front = Vec::new();
      // for each solution `p` in last front...
      for p_idx in last_front.iter() {
        // for each solution `q` dominated by `p`...
        for q_idx in dominance_lists[*p_idx].iter_mut() {
          // decrement counter of solutions dominating `q`
          dominance_counters[*q_idx] -= 1;
          // if no more solutions dominate `q`...
          if dominance_counters[*q_idx] == 0 {
            best_solutions_list[*q_idx] = true; // mark it as one of the best
            next_front.push(*q_idx); // and push its index into next front
          }
        }
      }
      front_solutions_count += next_front.len();
      last_front = next_front;
    }

    // calculate crowding distance for each solution in the last found front
    // for each objective `o`...
    for o_idx in 0..OBJ_CNT {
      // sort solutions by their scores of objective `o`
      last_front.sort_by(|&a_idx, &b_idx| {
        scores[a_idx][o_idx].total_cmp(&scores[b_idx][o_idx])
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
      let score_diff = if max_score - min_score != 0.0 {
        f64::from(max_score - min_score)
      } else {
        1.0
      };
      // for each solution except the first and the last in the last front...
      for solutions in last_front.windows(3) {
        let (prev_idx, curr_idx, next_idx) =
          (solutions[0], solutions[1], solutions[2]);
        let prev_cd = crowding_distances[prev_idx];
        let next_cd = crowding_distances[next_idx];
        // cd[i] = cd[i] + (cd[i+1] - cd[i-1]) / (max - min)
        crowding_distances[curr_idx] += (next_cd - prev_cd) / score_diff;
      }
    }

    // sort solutions in the last front by their crowding distances
    last_front.sort_by(|&a_idx, &b_idx| {
      crowding_distances[b_idx].total_cmp(&crowding_distances[a_idx])
    });
    // reset `best solution` flag for each excess solution in the last front
    for e_idx in last_front[front_solutions_count - solutions.len()..].iter() {
      best_solutions_list[*e_idx] = false;
    }

    let (new_sols, new_scs): (Vec<S>, Vec<Scores<OBJ_CNT>>) = solutions
      .into_iter()
      .zip(scores)
      .zip(best_solutions_list)
      .filter_map(|(sol_sc, is_best)| is_best.then_some(sol_sc))
      .unzip();

    assert_eq!(
      new_sols.len(),
      self.initial_population_size,
      "new population size must match initial population size"
    );
    assert_eq!(
      new_sols.len(),
      new_scs.len(),
      "number of solutions must match number of scores"
    );

    (new_sols, new_scs)
  }
}
