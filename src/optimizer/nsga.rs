use rayon::prelude::*;
use std::cmp::Ordering;
use std::hash::Hash;
use std::marker::PhantomData;

use crate::{
  objective::{pareto::ParetoDominance, Scores},
  Crossover, Mutator, Objectives, Optimizer, Selector, Solutions, Terminator,
};

struct SolutionData<S, const N: usize> {
  solution: S,
  scores: Scores<N>, // make it a reference?
}

impl<S, const N: usize> PartialEq for SolutionData<S, N> {
  fn eq(&self, other: &Self) -> bool {
    std::ptr::eq(self, other)
  }
}

impl<S, const N: usize> Eq for SolutionData<S, N> {}

impl<S, const N: usize> Hash for SolutionData<S, N> {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    std::ptr::hash(self, state);
  }
}

pub struct Nsga2<
  S,
  const OBJ_CNT: usize,
  const CRS_IN: usize,
  const CRS_OUT: usize,
  Obj: Objectives<OBJ_CNT, S>,
  Ter: Terminator<OBJ_CNT, S>,
  Sel: Selector<OBJ_CNT, S>,
  Crs: Crossover<CRS_IN, CRS_OUT, S>,
  Mut: Mutator<S>,
> {
  best_population: Vec<SolutionData<S, OBJ_CNT>>,
  new_population: Vec<S>,
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
    Sel: Selector<OBJ_CNT, S>,
    Crs: Crossover<CRS_IN, CRS_OUT, S>,
    Mut: Mutator<S>,
  > Optimizer<S> for Nsga2<S, OBJ_CNT, CRS_IN, CRS_OUT, Obj, Ter, Sel, Crs, Mut>
{
  fn run(self) {
    let mut this = self;
    let fresh_population = std::mem::take(&mut this.new_population);
    let fresh_population: Vec<_> = fresh_population
      .into_par_iter()
      .map(|solution| {
        let scores = this.objective.evaluate(&solution);
        SolutionData { solution, scores }
      })
      .collect();
  }
}

fn select_best_solutions<S, const N: usize>(
  population: Vec<SolutionData<S, N>>,
) -> Vec<SolutionData<S, N>> {
  // contains dominated solutions with their indicies by each solution
  let mut dominance_lists = vec![Vec::new(); population.len()];
  // contains number of solutions dominating each solution
  let mut dominance_counters = vec![0; population.len()];
  // if a solution belongs to the best ones, the corresponding value is `true`
  let mut best_solutions_list = vec![false; population.len()];
  let mut last_front = Vec::new();

  // fill dominance sets and counters
  for idx in 0..population.len() {
    // for each unique pair of solutions `p`...
    let (sd, rest) = population[idx..]
      .split_first()
      .expect("no solutions remain");
    // and `q`...
    for (j, other_sd) in rest.iter().enumerate() {
      let other_idx = idx + j + 1;
      match sd.scores.dominance(&other_sd.scores) {
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
      last_front.push((idx, sd)); // put it into the first front
    }
  }

  let mut front_solutions_count = last_front.len();
  // while last front isn't empty and we haven't found enough solutions...
  while !last_front.is_empty() && front_solutions_count < population.len() {
    let mut next_front = Vec::new();
    // for each solution `p` in last front...
    for (idx, _) in last_front.iter() {
      // for each solution `q` dominated by `p`...
      for (dommed_idx, dommed) in dominance_lists[*idx].iter_mut() {
        // decrement counter of solutions dominating `q`
        dominance_counters[*dommed_idx] -= 1;
        // if no more solutions dominate `q`...
        if dominance_counters[*dommed_idx] == 0 {
          best_solutions_list[*dommed_idx] = true; // mark it as one of the best
          next_front.push((*dommed_idx, *dommed)); // and push into next front
        }
      }
    }
    front_solutions_count += next_front.len();
    last_front = next_front;
  }

  // calculate crowding distance for the last found front
  // the idea is to remove boolean flags from `best_solutions_list` for
  // rejected solutions

  population
    .into_iter()
    .zip(best_solutions_list)
    .filter_map(|(s, is_best)| is_best.then_some(s))
    .collect()
}
