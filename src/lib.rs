//! **MOGA** is a Multi-Objective Genetic Algorithm framework for solving a
//! variety of multi-objective optimization problems. It strives to be simple,
//! performant and highly focused on usage of closures.
//!
//! Here's a [quick start example](#example) for the impatient.
//!
//! This crate provides an [`Optimizer`] abstraction that performs a common GA
//! loop using GA **operators**:
//! 1. **Select** solutions which are suitable for becoming parents of the next
//!    generation of solutions.
//! 2. **Recombine** selected solutions, creating the next generation of
//!    solutions.
//! 3. **Mutate** each solution.
//! 4. **Test** candidate solutions against certain objectives, evaluating
//!    fitness scores per each objective for a solution.
//! 5. **Terminate** the loop if a certain termination condition is met.
//!
//! There is a hidden step that occurs after evaluation of objective scores:
//! genetic algorithms usually truncate surplus solutions using some sort of
//! internally implemented truncation operator. Although, the implementation of
//! the loop itself may differ depending on algorithm.
//!
//! # Optimizers
//!
//! As for now, this crate features only one implememtation of [`Optimizer`] -
//! [`NSGA-II`]. It is a fast and simple genetic algorithm which you can read
//! about here:
//! <https://cs.uwlax.edu/~dmathias/cs419/readings/NSGAIIElitistMultiobjectiveGA.pdf>
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
//! whole arrays of solutions and their fitness values, another is applied to
//! solutions and scores individually or, in case of [`Recombination`], to small
//! groups of solutions. The table below lists those traits for each **operator**.
//!
//! | Operator                   | Applied to all solutions | Applied to each solution<br>or group of solutions |
//! |:--------------------------:|:------------------------:|:-------------------------------------------------:|
//! | **Test operator**          | [`Tester`]               | [`Test`]                                          |
//! | **Selection operator**     | [`Selector`]             | [`Selection`]                                     |
//! | **Recombination operator** | [`Recombinator`]         | [`Recombination`]                                 |
//! | **Mutation operator**      | [`Mutator`]              | [`Mutation`]                                      |
//! | **Termination operator**   | [`Terminator`]           | [`Termination`]                                   |
//!
//! Each pair of operators implements its **executor**. For example, you can use
//! [`Mutation`] instead of [`Mutator`] - **executor** will simply apply given
//! [`Mutation`] to each solution. Only [`Recombination`]s are a bit different:
//! they are applied to each *combination* of solutions.
//!
//! This crate does not provide the common crossover or mutation functions you'd
//! expect to see in a usual GA focused crate. The reason for this is that the
//! crate was developed to mainly operate not on numbers, but on structs or
//! *sets* of objects of unknown type. The crate does, however, provide a few
//! [selectors](crate::selection#structs) and one commonly used
//! [`GenerationTerminator`] for good measure.
//!
//! # Closures
//!
//! Each **operator** trait is implemented by respective closures. A [`Test`]
//! takes a solution of type `S` and returns an array of `f32` values - one
//! value per objective. Thus, instead of implementing [`Test`] you could use
//! a closure of type `Fn(&S) -> [f32; N]`. Consult operators' documentation
//! to see what closures implement which traits.
//!
//! Note, however, that this highly generic implementation leads to unreadable
//! compile time error messages that appear not during closure definition, but
//! during creation of an optimizer. If you are struggling with a closure, maybe
//! you should implement a trait directly instead. These implementations are
//! resolved during compilation, so neither approach is less performant.
//!
//! # Parallelization
//!
//! The **operators** from the 3rd column of the table above can be easily
//! parallelized by calling [`par_each()`] or [`par_batch()`] methods on them
//! (the latter isn't implemented for [`Recombination`]). This cheap conversion
//! only wraps the **operator** into a struct, tagging it, so an **executor**
//! will apply such **operator** in parallel to each solution/score or to their
//! batches of equal size. And you can call these methods on closures too:
//! ```
//! # use moga::operator::*;
//! let test = |f: &f32| [f + 1.0, f * 2.0];
//! let par_test = test.par_batch();
//! ```
//!
//! For simple operators, the overhead introduced by parallelization usually
//! only decreases performance, but when you need it, *you need it*. Benchmark,
//! if in doubt.
//!
//! # Example
//!
//! Below lies a solution for the textbook *Schaffer's Problem No.1*. This
//! solution is oversimplified and very suboptimal, but it demonstrates the
//! framework's workflow and manages to find Pareto optimal solutions for that
//! problem.
//! ```no_run
//! # use rand::{seq::IteratorRandom, Rng};
//! # use moga::{
//! #  optimizer::nsga::Nsga2,
//! #  selection::RandomSelector,
//! #  termination::GenerationTerminator,
//! #  Optimizer,
//! #  ParBatch,
//! # };
//! # fn main() {
//! // initial solutions lie between 0 and 100
//! let population = (0..100).map(|i| i as f32).collect::<Vec<_>>();
//! // objective functions `f1(x) = x^2` and `f2(x) = (x - 2)^2`
//! let test = |x: &f32| [x.powf(2.0), (x - 2.0).powf(2.0)];
//! // select 10 random solutions
//! let selector = RandomSelector(10);
//! // for each pair of parents `x` and `y` create an offspring
//! // `o = x + r * (y - x)` where `r` is a random value between -1 and 2
//! let r = || rand::thread_rng().gen_range(-1.0..2.0);
//! let recombinator = |x: &f32, y: &f32| x + r() * (y - x);
//! // don't mutate solutions
//! let mutation = |_: &mut f32| {};
//! // terminate after 100 generations
//! let terminator = GenerationTerminator(100);
//! // a convinient builder with compile time verification
//! let optimizer = Nsga2::builder()
//!   .population(population)
//!   // `test` will be executed concurrently for each batch of solutions
//!   .tester(test.par_batch())
//!   .selector(selector)
//!   .recombinator(recombinator)
//!   .mutator(mutation)
//!   .terminator(terminator)
//!   .build();
//! // upon termination the optimizer returns the best solutions it has found
//! let solutions = optimizer.optimize();
//! # }
//! ```
//!
//! You can find more examples in the *examples* folder in the root of the
//! project. You can also write one yourself and contribute it to the crate. I'd
//! be very grateful! Here is a list of functions that one day I hope to cover:
//! <https://en.wikipedia.org/wiki/Test_functions_for_optimization>
//!
//! # Common pitfalls
//!
//! - Closures are great and handy to use until they aren't. A subtle mistake
//!   can paint your code red and the error will appear far away from where you
//!   actually made a mistake. Since Rust does not allow you to annotate your
//!   variables with traits, always keep an eye on your closures or just
//!   implement traits for your own types instead.
//!
//! - [`Selection`] and [`Termination`] are implemented for the same closure of
//!   type `Fn(&S, &[f32; N]) -> bool` which may confuse the compiler (and you)
//!   from time to time. Move closures into an optimizer as soon as possible,
//!   or, again, implement those traits for your own type.
//!
//! - More often than not, parallelization only decreases performance of the
//!   algorithm. Currently, Rust does not provide any benchmarking utilities
//!   "out-of-the-box" but you can use the tools that your OS has, like
//!   [time] on Linux or [Measure-Command] in PowerShell on Windows. Make the
//!   number of generations low and test it.
//!
//! - Types may be hard to keep track of. For example, your solution candidate
//!   can be of type `(f32, f32)` and if you optimize for two objectives then
//!   you'll end up with signatures like `|_: &[(f32, f32)], _: &[[f32, f32]]|`
//!   which leads to hard-to-catch errors once you mix up something.
//!   This also applies to the number of objectives: if your `test` function
//!   produces 2 values per solutions but a `selector` expects 3, then you'll
//!   get a compile-time  error. To avoid this, use [type aliases]. This crate
//!   itself uses aliases defined in [`score`] module. And you can use them too.
//!
//! # Use cases
//!
//! *This crate was designed to solve a very specific problem which, in case of*
//! *success, will certainly appear in this list. If this crate happens to be*
//! *useful to you, please contact me and I'll put your repo on the list.*
//!
//! [`Optimizer`]: crate::optimizer::Optimizer
//! [`NSGA-II`]: crate::optimizer::nsga::Nsga2
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
//! [time]: https://www.man7.org/linux/man-pages/man1/time.1.html
//! [Measure-Command]: https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/measure-command
//! [type aliases]: https://doc.rust-lang.org/reference/items/type-aliases.html

#![warn(missing_docs)]

mod execution;
pub mod mutation;
pub mod operator;
pub mod optimizer;
pub mod recombination;
pub mod score;
pub mod selection;
pub mod termination;
pub mod testing;

// common operators and traits are re-exported by the crate. the less common
// ones, along with optimizer implementations, must be imported from their
// respective modules.
pub use self::{
  mutation::{Mutation, Mutator},
  operator::{ParBatch, ParBatchOperator, ParEach, ParEachOperator},
  optimizer::Optimizer,
  recombination::{Recombination, Recombinator},
  score::{Score, Scores},
  selection::{Selection, Selector},
  termination::{Termination, Terminator},
  testing::{Test, Tester},
};
