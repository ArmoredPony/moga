//! Binh and Korn problem solution using SPEA-II.

use moga::{
  operator::{ParBatch, ParEach},
  optimizer::{spea::Spea2, Optimizer},
  selection::TournamentSelectorWithoutReplacement,
  termination::GenerationTerminator,
};
use rand::Rng;
use rand_distr::{Distribution, Normal};

fn main() {
  // the 'Solution' type represented by a pair of floating point values
  struct Solution {
    x: f32,
    y: f32,
  }

  // the initial population
  let population: Vec<Solution> = (0i8..100)
    .map(|i| Solution {
      x: i.into(),
      y: i.into(),
    })
    .collect();

  // the archive size of `Spea2` optimizer
  let archive_size = population.len();

  // the second objective function f1(x, y) = 4x^2 + 4y^2
  let f1 = |s: &Solution| 4.0 * s.x.powf(2.0) + 4.0 * s.y.powf(2.0);
  // and the second objective function f2(x, y) = (x - 5)^2 + (y - 5)^2
  let f2 = |s: &Solution| (s.x - 5.0).powf(2.0) + (s.y - 5.0).powf(2.0);

  // an array of closures forms a `Test`
  let _test = [f1, f2];

  // you can also create a `Test` from a closure that returns an array
  // instead of using array of closures
  let test = |s: &Solution| {
    [
      4.0 * s.x.powf(2.0) + 4.0 * s.y.powf(2.0),
      (s.x - 5.0).powf(2.0) + (s.y - 5.0).powf(2.0),
    ]
  };

  // a `Terminator` that terminates after 1000 generations
  let terminator = GenerationTerminator(1000);

  // a `Selector` that selects 10 unique solutions from 10 binary tournaments
  let selector = TournamentSelectorWithoutReplacement(10, 2);

  // simulated binary crossover for two `f32` values...
  let sbx_f32 = |a: f32, b: f32| -> (f32, f32) {
    let n = 2.0;
    let r: f32 = rand::thread_rng().gen_range(0.0..1.0);
    let beta = if r <= 0.5 {
      (2.0 * r).powf(1.0 / (n + 1.0))
    } else {
      (1.0 / (2.0 * (1.0 - r))).powf(1.0 / (n + 1.0))
    };
    let p = 0.5 * ((a + b) - beta * (b - a));
    let q = 0.5 * ((a + b) + beta * (b - a));
    (p, q)
  };
  // which is applied to both solutions' values by `Recombination` operator
  let recombination = |a: &Solution, b: &Solution| -> (Solution, Solution) {
    let (x1, x2) = sbx_f32(a.x, b.x);
    let (y1, y2) = sbx_f32(a.y, b.y);
    (Solution { x: x1, y: y1 }, Solution { x: x2, y: y2 })
  };

  // a `Mutator` based on random values from normal disribution...
  let normal = Normal::new(0.0, 1.0).unwrap(); // which comes from 'rand_distr'
  let mutation = |s: &mut Solution| {
    s.x += normal.sample(&mut rand::thread_rng());
    s.y += normal.sample(&mut rand::thread_rng());
  };

  // a convinient builder with compile time verification from `typed-builder` crate
  let spea2 = Spea2::builder()
    .population(population)
    .archive_size(archive_size)
    // fitness values will be evaluated concurrently for each solution
    .tester(test.par_each())
    .selector(selector)
    .recombinator(recombination)
    // solutions will be split in batches of optimal size, then the mutation
    // operator will be applied to solutions in each batch concurrently
    .mutator(mutation.par_batch())
    .terminator(terminator)
    .build();

  // consume and run the optimizer, returning the best solutions
  let solutions = spea2.optimize();

  // print values of objective functions for each solution
  for s in solutions {
    let [x, y] = test(&s);
    println!("{x} {y}",);
  }
}
