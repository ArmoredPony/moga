use moga::*;
use rand::prelude::*;

fn main() {
  // our 'solution' type represented by a pair of floating point valeus
  type S = (f32, f32);

  // initial population
  let population: Vec<S> = (0i8..100).map(|i| (i.into(), i.into())).collect();

  // objective function f1(x, y) = 4x^2 + 4y^2
  let f1 = |&(a, b): &(f32, f32)| 4.0 * a.powf(2.0) + 4.0 * b.powf(2.0);
  // and another objective function f2(x, y) = (x - 5)^2 + (y - 5)^2
  let f2 = |&(a, b): &(f32, f32)| (a - 5.0).powf(2.0) + (b - 5.0).powf(2.0);
  let objectives = [f1, f2];

  // terminates after 1000 generations
  let terminator = GenerationsTerminator(100);

  // selects 10 values randomly
  let selector = RandomSelector(10, rand::thread_rng());

  // SBX crossover for two floating point values
  let f32_crossover = |a: f32, b: f32| -> (f32, f32) {
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
  // ...which is used on both solutions' values
  let crossover = |(x1, y1): &S, (x2, y2): &S| -> (S, S) {
    let (x3, x4) = f32_crossover(*x1, *x2);
    let (y3, y4) = f32_crossover(*y1, *y2);
    ((x3, y3), (x4, y4))
  };

  // and no mutator
  let mutator = NoMutator();

  let nsga = Nsga2::new(
    population, objectives, terminator, selector, crossover, mutator,
  );
  let results = nsga.run();
  print!("l=[");
  let mut iter = results.into_iter();
  let (x, y) = iter.next().unwrap();
  print!("({x:2},{y:2})");
  for (x, y) in iter {
    print!(",({x:2},{y:2})");
  }
  print!("]");
}
