use std::{cmp::Ordering, marker::PhantomData};

use rayon::prelude::*;

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
  Ter: Terminator<OBJ_CNT, S>,
  Sel: Selector<S, OBJ_CNT>,
  Crs: Crossover<CRS_IN, CRS_OUT, S>,
  Mut: Mutator<S>,
> {
  best_solutions: Vec<(S, Scores<OBJ_CNT>)>,
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
    S: Send,
    const OBJ_CNT: usize,
    const CRS_IN: usize,
    const CRS_OUT: usize,
    Obj: Objectives<OBJ_CNT, S> + Sync,
    Ter: Terminator<OBJ_CNT, S>,
    Sel: Selector<S, OBJ_CNT>,
    Crs: Crossover<CRS_IN, CRS_OUT, S>,
    Mut: Mutator<S>,
  > Optimizer<S>
  for Nsga2<S, OBJ_CNT, CRS_IN, CRS_OUT, Obj, Ter, Sel, Crs, Mut>
{
  fn run(mut self) -> Vec<S> {
    self.best_solutions = std::mem::take(&mut self.new_solutions)
      .into_par_iter()
      .map(|solution| {
        let scores = self.objective.evaluate(&solution);
        (solution, scores)
      })
      .collect();

    while !self.terminator.terminate(&self.best_solutions) {
      let selected_solutions = self.selector.select(&self.best_solutions);
      let created_solutions = self.crossover.create(&selected_solutions);
      let mut created_solutions_scores: Vec<_> = created_solutions
        .into_iter()
        .map(|solution| {
          let scores = self.objective.evaluate(&solution);
          (solution, scores)
        })
        .collect();
      created_solutions_scores
        .iter_mut()
        .for_each(|(s, _)| self.mutator.mutate(s));
      let mut all_solutions = std::mem::take(&mut self.best_solutions);
      all_solutions.append(&mut created_solutions_scores);
      self.best_solutions = self.select_best_solutions(all_solutions);
    }

    self.best_solutions.into_iter().map(|(s, _)| s).collect()
  }
}

impl<
    S: Send,
    const OBJ_CNT: usize,
    const CRS_IN: usize,
    const CRS_OUT: usize,
    Obj: Objectives<OBJ_CNT, S> + Sync,
    Ter: Terminator<OBJ_CNT, S>,
    Sel: Selector<S, OBJ_CNT>,
    Crs: Crossover<CRS_IN, CRS_OUT, S>,
    Mut: Mutator<S>,
  > Nsga2<S, OBJ_CNT, CRS_IN, CRS_OUT, Obj, Ter, Sel, Crs, Mut>
{
  // really really really want to refactor this mess
  fn select_best_solutions(
    &mut self,
    population: Vec<(S, Scores<OBJ_CNT>)>,
  ) -> Vec<(S, [f32; OBJ_CNT])> {
    // contains dominated solutions with their indicies by each solution
    let mut dominance_lists = vec![Vec::new(); population.len()];
    // contains number of solutions dominating each solution
    let mut dominance_counters = vec![0; population.len()];
    // if a solution belongs to the best ones, the corresponding value is `true`
    let mut best_solutions_list = vec![false; population.len()];
    let mut first_front = Vec::new();

    // fill dominance sets and counters
    for idx in 0..population.len() {
      // for each unique pair of solutions `p`...
      let (sd, rest) = population[idx..]
        .split_first()
        .expect("no solutions remain");
      // and `q`...
      for (j, other_sd) in rest.iter().enumerate() {
        let other_idx = idx + j + 1;
        match sd.1.dominance(&other_sd.1) {
          // if solution `p` dominates solution `q`...
          Ordering::Less => {
            // put solution `q` into list of solutions dominated by `p`
            dominance_lists[idx].push((other_idx, other_sd));
            // and increment counter of solutions dominating `q`
            dominance_counters[other_idx] += 1;
          }
          // if solution `q` dominates solution `p`...
          Ordering::Greater => {
            // put solution `p` into list of solutions dominated by `q`
            dominance_lists[other_idx].push((idx, sd));
            // and increment counter of solutions dominating `p`
            dominance_counters[idx] += 1;
          }
          Ordering::Equal => {}
        }
      }
      // if solution `p` isn't dominated by any other solution...
      if dominance_counters[idx] == 0 {
        first_front.push((idx, sd, 0.0f64)); // put it into the first front
      }
    }

    let mut last_front = first_front;
    let mut front_solutions_count = last_front.len();
    // while last front isn't empty and we haven't found enough solutions...
    while !last_front.is_empty() && front_solutions_count < population.len() {
      let mut next_front = Vec::new();
      // for each solution `p` in last front...
      for (idx, ..) in last_front.iter() {
        // for each solution `q` dominated by `p`...
        for (dommed_idx, dommed, ..) in dominance_lists[*idx].iter_mut() {
          // decrement counter of solutions dominating `q`
          dominance_counters[*dommed_idx] -= 1;
          // if no more solutions dominate `q`...
          if dominance_counters[*dommed_idx] == 0 {
            best_solutions_list[*dommed_idx] = true; // mark it as one of the best
            next_front.push((*dommed_idx, *dommed, 0.0f64)); // and push into next front
          }
        }
      }
      front_solutions_count += next_front.len();
      last_front = next_front;
    }

    // calculate crowding distance for the last found front
    for obj_idx in 0..OBJ_CNT {
      last_front.sort_by(|a, b| a.1 .1[obj_idx].total_cmp(&b.1 .1[obj_idx]));
      let mut first_solution = last_front[0];
      let mut last_solution = last_front[last_front.len() - 1];
      // set crowding distances of first and last front member to `f64::MAX`
      first_solution.2 = f64::MAX;
      last_solution.2 = f64::MAX;
      // calculate difference between max and min score of current objective
      let min_score = first_solution.1 .1[obj_idx];
      let max_score = last_solution.1 .1[obj_idx];
      let score_diff = if max_score - min_score != 0.0 {
        f64::from(max_score - min_score)
      } else {
        1.0
      };
      // for each solution except the first and the last in the last front...
      for solutions in last_front.windows(3) {
        let prev_sol_score = solutions[0].1 .1[obj_idx]; // cd[i-1]
        let next_sol_score = solutions[2].1 .1[obj_idx]; // cd[i+1]
        let mut cur_solution = solutions[1];
        // cd[i] = cd[i] + (cd[i+1] - cd[i-1]) / (max - min)
        cur_solution.2 +=
          f64::from(next_sol_score - prev_sol_score) / score_diff
      }
    }

    // sort solutions in the last front by their crowding distances
    last_front.sort_by(|a, b| b.2.total_cmp(&a.2));
    // reset `best solution` flag for each excess solution in the last front
    for excess in last_front[front_solutions_count - population.len()..].iter()
    {
      best_solutions_list[excess.0] = false;
    }

    let new_population: Vec<_> = population
      .into_iter()
      .zip(best_solutions_list)
      .filter_map(|(s, is_best)| is_best.then_some(s))
      .collect();
    debug_assert_eq!(
      new_population.len(),
      self.initial_population_size,
      "selected population size must match initial population size"
    );
    new_population
  }
}
