//! Binh and Korn problem solution using SPEA-II.

#![allow(unused_variables)]

use moga::{
  operator::{ParBatch, ParEach},
  optimizer::{spea::Spea2, Optimizer},
  selection::TournamentSelectorWithoutReplacement,
  termination::GenerationTerminator,
};
use rand::Rng;
use rand_distr::{Distribution, Normal};

fn main() {
  // our 'solution' type represented by a pair of floating point valeus
  type S = (f32, f32);

  // initial population
  let population: Vec<S> = (0i8..100).map(|i| (i.into(), i.into())).collect();

  // archive size of `Spea2` optimizer
  let archive_size = population.len();

  // objective function f1(x, y) = 4x^2 + 4y^2
  let f1 = |&(a, b): &S| 4.0 * a.powf(2.0) + 4.0 * b.powf(2.0);
  // and another objective function f2(x, y) = (x - 5)^2 + (y - 5)^2
  let f2 = |&(a, b): &S| (a - 5.0).powf(2.0) + (b - 5.0).powf(2.0);

  // array of closures forms a `Test`
  let test = [f1, f2];

  // you can also create a `Test` from a closure that returns an array
  // instead of using array of closures
  let test = |&(a, b): &S| {
    [
      4.0 * a.powf(2.0) + 4.0 * b.powf(2.0),
      (a - 5.0).powf(2.0) + (b - 5.0).powf(2.0),
    ]
  };

  // a `Terminator` that terminates after 1000 generations
  let terminator = GenerationTerminator(1000);

  // a `Selector` that selects 10 unique solutions from 10 binary tournaments
  let selector = TournamentSelectorWithoutReplacement(10, 2);

  // simulated binary crossover for two `f32` values...
  let sbx_f32 = |a: f32, b: f32| -> S {
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
  let recombination = |(x1, y1): &S, (x2, y2): &S| -> (S, S) {
    let (x3, x4) = sbx_f32(*x1, *x2);
    let (y3, y4) = sbx_f32(*y1, *y2);
    ((x3, y3), (x4, y4))
  };

  // a `Mutator` based on random values from normal disribution...
  let normal = Normal::new(0.0, 1.0).unwrap(); // which comes from 'rand_distr'
  let mutation = |s: &mut S| {
    s.0 += normal.sample(&mut rand::thread_rng());
    s.1 += normal.sample(&mut rand::thread_rng());
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
