//! Schaffer's Problem No.1 solution using NSGA-II.

use moga::{
  operator::ParBatch,
  optimizer::{nsga::Nsga2, Optimizer},
  selection::RandomSelector,
  termination::GenerationTerminator,
};
use rand::Rng;

fn main() {
  // initial solutions lie between 0 and 100
  let population = (0..100).map(|i| i as f32).collect::<Vec<_>>();

  // objective functions `f1(x) = x^2` and `f2(x) = (x - 2)^2`
  let test = |x: &f32| [x.powf(2.0), (x - 2.0).powf(2.0)];

  // a `Selector` that selects 10 random solutions
  let selector = RandomSelector(10);

  // for each pair of parents `x` and `y` create an offspring
  // `o = x + r * (y - x)` where `r` is a random value between -1 and 2
  let r = || rand::thread_rng().gen_range(-1.0..2.0);
  let recombinator = |x: &f32, y: &f32| x + r() * (y - x);

  // a `Mutation` that does not mutate solutions
  let mutation = |_: &mut f32| {};

  // a `Termiantor` that terminates after 100 generations
  let terminator = GenerationTerminator(100);

  // a convinient builder with compile time verification from `typed-builder` crate
  let nsga2 = Nsga2::builder()
    .population(population)
    // `test` will be executed concurrently for each batch of solutions
    .tester(test.par_batch())
    .selector(selector)
    .recombinator(recombinator)
    .mutator(mutation)
    .terminator(terminator)
    .build();

  // upon termination optimizer returns the best solutions it has found
  let solutions = nsga2.optimize().unwrap();

  // print values of objective functions for each solution
  for s in solutions {
    let [x, y] = test(&s);
    println!("{x} {y}",);
  }
}
