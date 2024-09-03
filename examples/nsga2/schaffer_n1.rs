//! Schaffer's Problem No.1 solution using NSGA-II.

use std::{fs::File, io::Write, path::Path};

use moga::{
  optimizer::nsga::Nsga2,
  selection::RandomSelector,
  termination::GenerationTerminator,
  Optimizer,
  ParBatch,
};
use rand::{seq::IteratorRandom, Rng};

fn main() {
  // initial solutions lie between 0 and 100
  let population = (0..100).map(|i| i as f32).collect::<Vec<_>>();
  // objective functions `f1(x) = x^2` and `f2(x) = (x - 2)^2`
  let test = |x: &f32| [x.powf(2.0), (x - 2.0).powf(2.0)];
  // select 10 random solutions
  let selector = RandomSelector(10);
  // for each pair of parents `x` and `y` create an offspring
  // `o = x + r * (y - x)` where `r` is a random value between -1 and 2
  let r = || rand::thread_rng().gen_range(-1.0..2.0);
  let recombinator = |x: &f32, y: &f32| x + r() * (y - x);
  // don't mutate solutions
  let mutation = |_: &mut f32| {};
  // terminate after 100 generations
  let terminator = GenerationTerminator(100);

  // a convinient builder with compile time verification from `typed-builder` crate
  let optimizer = Nsga2::builder()
    .population(population)
    // `test` will be executed concurrently for each batch of solutions
    .tester(test.par_batch())
    .selector(selector)
    .recombinator(recombinator)
    .mutator(mutation)
    .terminator(terminator)
    .build();

  // upon termination optimizer returns the best solutions it has found
  let solutions = optimizer.optimize();

  // write solutions to file in examples/nsga2/binh_korn.csv
  let _ = File::create(Path::new(file!()).with_file_name("schaffer_n1.csv"))
    .unwrap()
    .write_all(
      solutions
        .iter()
        .map(|x| {
          let [a, b] = test(x);
          format!("{} {}", a, b)
        })
        .collect::<Vec<_>>()
        .join("\n")
        .as_bytes(),
    );

  // print 10 random solutions
  for x in solutions
    .into_iter()
    .choose_multiple(&mut rand::thread_rng(), 10)
  {
    println!("{x:.4}");
  }
  println!("  ...  ");
}
