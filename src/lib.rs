//! **MOGA** is a Multi-Objective Genetic Algorithm framework for solving
//! various multi-objective optimization problems. It strives to be simple,
//! performant and highly focused on usage of closures.
//!
//! Here's a [quick start example](#example) for the impatient.
//!
//! This crate defines a few abstractions that allow for flexible construction
//! of genetic algorithms. These abstractions aren't necessarily represented in
//! crate's user API, as they serve the purpose of helping you to understand the
//! workflow of this framework.
//! - **Operator** - is an abstraction over genetic operators: **selection**,
//!   **recombination**, **mutation**, **test**, and **termination**
//! - Each **operator** is executed by a respective **executor** - an internal
//!   abstraction, that executes operators based on their type and execution
//!   strategy: **sequential** or **parallel**
//! - **Optimizer** is an abstraction that controls execution of each step of
//!   a typical genetic algorithm loop:
//!   1. **Select** solutions which are suitable for becoming parents of the
//!      next generation of solutions
//!   2. **Recombine** selected solutions, creating the next generation of
//!      solutions
//!   3. **Mutate** each solution
//!   4. **Test** solutions against certain objectives, evaluating fitness
//!      scores per each objective for each solution
//!   5. **Terminate** the loop if a certain termination condition is met
//!
//! In general, a user should simply supply five genetic **operators** to an
//! implementation of **optimizer**, which executes them with **executors**,
//! performing the GA loop and eventually finding Pareto optimal solutions for
//! the posed problem.
//!
//! # Optimizers
//!
//! **Optimizer** is an abstraction represented in this crate with [`Optimizer`]
//! trait. Usually, an **optimizer** consumes five - one per **operator** -
//! **executors**, which are supplied by the user, and runs the GA loop until
//! it reaches some termination condition. Although, the implementation of this
//! loop may differ depending on algorithm.
//!
//! As for now, this crate features two implementations of [`Optimizer`] -
//! [NSGA-II] and [SPEA-II].
//!
//! If you happen to implement another kind of genetic algorithm using this
//! framework, please contribute. The more options one has - the better.
//!
//! # Operators and Executors
//!
//! Abstractions, that perform each step, are called **operators**. For each
//! **operator**, this crate provides an **executor** - crate's internal
//! abstraction, that controls application of each operator to solutions and
//! scores.
//!
//! Each **operator** is represented with two traits. One of them operates on
//! whole arrays of solutions and their fitness scores, another is applied to
//! solutions and scores individually or, in case of [`Recombination`], to small
//! groups of solutions. The table below lists those traits for each **operator**.
//!
//! |                            | Applied to all solutions | Applied to each solution<br>or group of solutions |
//! |:---------------------------|:------------------------:|:-------------------------------------------------:|
//! | **Selection operator**     | [`Selector`]             | [`Selection`]                                     |
//! | **Recombination operator** | [`Recombinator`]         | [`Recombination`]                                 |
//! | **Mutation operator**      | [`Mutator`]              | [`Mutation`]                                      |
//! | **Test operator**          | [`Tester`]               | [`Test`]                                          |
//! | **Termination operator**   | [`Terminator`]           | [`Termination`]                                   |
//!
//! Each pair of operators implements its respective **executor**. For example,
//! you can use [`Mutation`] instead of [`Mutator`] when building an
//! **optimizer** - **executor** will simply apply given [`Mutation`] to each
//! solution. Only [`Recombination`]s are a bit different: they are applied to
//! each [combination] of solutions.
//!
//! This crate does not provide the common crossover or mutation functions you'd
//! expect to see in a usual GA focused crate. The reason for this is that
//! **genetic algorithms must be tailored to each problem to solve**. Fail to do
//! that - and this crate will do no better than random search. A predefined set
//! of operators will almost certainly prompt you to choose a less suitable, but
//! ready-made option. When working with genetic algorithms, it's too easy to
//! fall into this trap, conclude that genetic algorithms are not effective and
//! give up on them.
//!
//! However, this crate does implement for you a few
//! [selectors](crate::selection#structs) and one commonly used
//! [`GenerationTerminator`] for good measure.
//!
//! # Closures
//!
//! Each **operator** trait is implemented by one or several closures. For
//! example, a [`Test`] takes a reference to a solution of type `S` and returns
//! an array of `f32` values - one value per objective. Thus, instead of
//! implementing the [`Test`] trait for some struct, you can just create a
//! closure of type `Fn(&S) -> [f32; N]`. Consult the *Implementors* section of
//! operators' documentation to see what closures implement them, or check out
//! the *Examples*. You can navigate to each operator using the table from the
//! previous section.
//!
//! Note, however, that this highly generic implementation leads to unreadable
//! compiler error messages that appear not at closure definition, but at
//! creation of an optimizer. If you are struggling with a closure, maybe
//! you should implement a trait directly instead. These implementations are
//! resolved during compilation, so neither approach is less performant.
//!
//! # Parallelization
//!
//! The **operators** from the 2nd column of the table above can be easily
//! parallelized by calling [`par_each()`] or [`par_batch()`] methods on them
//! (the latter isn't implemented for [`Recombination`]). This cheap conversion
//! only wraps the **operator** into a struct, tagging it, so an **executor**
//! will apply such **operator** in parallel to each solution/score or to their
//! batches of equal size. And you can call these methods on closures too:
//! ```
//! # use moga::{operator::ParBatch, optimizer::nsga::Nsga2, score::Scores};
//! let test = |f: &f32| [f + 1.0, f * 2.0];
//! let par_test = test.par_batch();
//! let optimizer = Nsga2::builder()
//! #   .population(vec![])
//! #   .selector(|_: &f32, _: &Scores<2>| true)
//! #   .recombinator(|_: &f32, _: &f32| 0.0)
//! #   .mutator(|_: &mut f32| {})
//! #   .terminator(|_: &f32, _: &Scores<2>| false)
//!   // ...
//!   .tester(par_test)
//!   // ...
//!   .build();
//! ```
//!
//! Note that to be parallelized, both **operators** and solutions must
//! implement `Sync` or `Sync + Send`. Here is a table of traits that each
//! **operator** and its solution must implement to be parallelized.
//!
//! |                   | Operator must implement | Solution must implement |
//! |:------------------|:-----------------------:|:-----------------------:|
//! | [`Selection`]     | `Sync`                  | `Sync`                  |
//! | [`Recombination`] | `Sync`                  | `Sync + Send`           |
//! | [`Mutation`]      | `Sync`                  | `Sync + Send`           |
//! | [`Test`]          | `Sync`                  | `Sync`                  |
//! | [`Termination`]   | `Sync`                  | `Sync`                  |
//!
//! For simple operators, the overhead introduced by parallelization usually
//! only decreases performance, but when you need it, *you need it*. Benchmark,
//! if in doubt.
//!
//! # Example
//!
//! Here's a solution for the textbook *Schaffer's Problem No.1* with the
//! [SPEA-II] optimizer. This solution is oversimplified and very suboptimal,
//! but it demonstrates the framework's workflow and manages to find Pareto
//! optimal solutions for that problem.
//! ```no_run
//! # fn main() {
//! use moga::{
//!   operator::ParBatch,
//!   optimizer::{spea::Spea2, Optimizer},
//!   selection::RouletteSelector,
//!   termination::GenerationTerminator,
//! };
//! use rand::Rng;
//! // initial solutions lie between 0 and 100
//! let population = (0..100).map(|i| i as f32).collect::<Vec<_>>();
//! // archive size of `Spea2` optimizer
//! let archive_size = 100;
//! // objective functions `f1(x) = x^2` and `f2(x) = (x - 2)^2`
//! let test = |x: &f32| [x.powf(2.0), (x - 2.0).powf(2.0)];
//! // a `Selector` that selects 10 random solutions. selection chance of a
//! // solution is directly proportional to the number of solutions it dominates
//! let selector = RouletteSelector(10);
//! // for each pair of parents `x` and `y` create an offspring
//! // `o = x + r * (y - x)` where `r` is a random value between -1 and 2
//! let r = || rand::thread_rng().gen_range(-1.0..2.0);
//! let recombinator = |x: &f32, y: &f32| x + r() * (y - x);
//! // a `Mutation` that does not mutate solutions
//! let mutation = |_: &mut f32| {};
//! // a `Termiantor` that terminates after 100 generations
//! let terminator = GenerationTerminator(100);
//! // a convinient builder with compile time verification from `typed-builder` crate
//! let spea2 = Spea2::builder()
//!   .population(population)
//!   .archive_size(archive_size)
//!   // `test` will be executed concurrently for each batch of solutions
//!   .tester(test.par_batch())
//!   .selector(selector)
//!   .recombinator(recombinator)
//!   .mutator(mutation)
//!   .terminator(terminator)
//!   .build();
//! // upon termination optimizer returns the best solutions it has found
//! let solutions = spea2.optimize().unwrap();
//! # }
//! ```
//!
//! You can find more examples in the *examples* folder in the root of the
//! project. You can also write one yourself and contribute it to the crate. I'd
//! be very grateful! Here is a list of functions that I hope to have covered
//! one day:
//! <https://en.wikipedia.org/wiki/Test_functions_for_optimization>
//!
//! # Common pitfalls
//!
//! - Closures are great and handy to use until they aren't. A subtle mistake
//!   can paint your code red and the error will appear far away from where you
//!   actually made a mistake. Since Rust does not allow you to annotate your
//!   variables with traits, always keep an eye on your closures or just
//!   implement traits for your own types instead.
//! - [`Selection`] and [`Termination`] traits are implemented for the same
//!   closure of type `Fn(&S, &[f32; N]) -> bool` which may confuse the compiler
//!   (and you) from time to time. Move closures into an optimizer as soon as
//!   possible, or, again, implement those traits for your own type.
//! - More often than not, parallelization only decreases performance of the
//!   algorithm. Currently, Rust does not provide any benchmarking utilities
//!   "out-of-the-box", but you can use the tools that your OS has, like
//!   [time] on Linux or [Measure-Command] in PowerShell on Windows. Or even
//!   better, [timeit] from cross-platform shell [Nushell]. Make the number of
//!   generations low and test it.
//!
//! [`Optimizer`]: crate::optimizer::Optimizer
//! [NSGA-II]: https://sci2s.ugr.es/sites/default/files/files/Teaching/OtherPostGraduateCourses/Metaheuristicas/Deb_NSGAII.pdf
//! [SPEA-II]: https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/145755/eth-24689-01.pdf
//! [`Tester`]: crate::testing::Tester
//! [`Test`]: crate::testing::Test
//! [`Selector`]: crate::selection::Selector
//! [`Selection`]: crate::selection::Selection
//! [`Recombinator`]: crate::recombination::Recombinator
//! [`Recombination`]: crate::recombination::Recombination
//! [`Mutator`]: crate::mutation::Mutator
//! [`Mutation`]: crate::mutation::Mutation
//! [`Terminator`]: crate::termination::Terminator
//! [`Termination`]: crate::termination::Termination
//! [`GenerationTerminator`]: crate::termination::GenerationTerminator
//! [`par_each()`]: crate::operator::ParEach::par_each
//! [`par_batch()`]: crate::operator::ParBatch::par_batch
//! [`score`]: crate::score
//! [combination]: https://en.wikipedia.org/wiki/Combination
//! [time]: https://www.man7.org/linux/man-pages/man1/time.1.html
//! [Measure-Command]: https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/measure-command
//! [timeit]: https://www.nushell.sh/commands/docs/timeit.html
//! [Nushell]: https://www.nushell.sh/

#![warn(missing_docs)]

pub mod constraining;
mod execution;
pub mod mutation;
pub mod operator;
pub mod optimizer;
pub mod recombination;
pub mod score;
pub mod selection;
pub mod termination;
pub mod testing;
