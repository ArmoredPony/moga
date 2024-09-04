# MOGA

**MOGA** is a Multi-Objective Genetic Algorithm framework for solving a variety
of multi-objective optimization problems. It strives to be simple, performant
and highly focused on usage of closures.

This crate provides you with five genetic operator abstractions that you can
implement and insert into an optimizer - another abstraction, that will run the
common genetic algorithm loop using your operators:

1. **Select** solutions which are suitable for becoming parents of the next
   generation of solutions.
2. **Recombine** selected solutions, creating the next generation of solutions.
3. **Mutate** each solution.
4. **Test** candidate solutions against certain objectives, evaluating fitness
   scores per each objective for a solution.
5. **Terminate** the loop if a certain termination condition is met.

Each operator can be implemented with a closure and optionally parallelized
with [rayon](https://crates.io/crates/rayon) by adding just a single method
call.

This crate and each its module features rich [documentation](https://docs.rs/moga).
Read at least some of it. Or jump straight to the [example](#example) and start
writing your own code.

## Features

- Convenient abstractions over genetic operators that are executed by
  optimizers
- Optional and easily achievable parallelization of application of your
  operators backed by famous [rayon](https://crates.io/crates/rayon) crate
- Closures as trait implementors ~~almost~~ everywhere you like
- Highly generic code with absolutely unreadable compiler error messages should
  you make a mistake somewhere
- Not two, not three, but **one** implementation of a genetic algorithm -
  [NSGA-II](https://cs.uwlax.edu/~dmathias/cs419/readings/NSGAIIElitistMultiobjectiveGA.pdf)
- Everything is documented! Just read the [docs](https://docs.rs/moga)

## Example

Here's a solution for the textbook *Schaffer's Problem No.1*. This solution
is oversimplified and very suboptimal, but it demonstrates the framework's
workflow and manages to find Pareto optimal solutions for that problem.
```rust
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
// a convinient builder with compile time verification
let optimizer = Nsga2::builder()
  .population(population)
  // `test` will be executed concurrently for each batch of solutions
  .tester(test.par_batch())
  .selector(selector)
  .recombinator(recombinator)
  .mutator(mutation)
  .terminator(terminator)
  .build();
// upon termination the optimizer returns the best solutions it has found
let solutions = optimizer.optimize();
```

## Use cases

*This crate was designed to solve a very specific problem which, in case of
success, will certainly appear in this list. If this crate happens to be
useful to you, please contact me, and I'll be happy to put your repo on the
list.*

## Contributions

Contributions are very welcome, be it another genetic algorithm implementation
or an example of some problem solved with **MOGA**. Don't forget to write some
docs and tests, and run *rustfmt* and *clippy* on your code.

## License

This crate is licensed under [MIT](./LICENSE) license.