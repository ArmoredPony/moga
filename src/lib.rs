//! **MOGA** is a Multi-Objective Genetic Algorithm framework for solving a
//! variety of multi-purpose optimization problems. It strives to be simple,
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
//! There is a hidden step that occures after evaluation of objective scores:
//! genetic algorithms usually truncate surplus solutions using some sort of
//! internally implemented truncation operator. Although, the implementation of
//! main GA loop itself may differ depending on algorithm.
//!
//! # Optimizers
//!
//! As for now, this crate features only one implememtation of [`Optimizer`] -
//! [`NSGA-II`]. It is a fast and simple genetic algorithm which you can read about here:
//! <https://cs.uwlax.edu/~dmathias/cs419/readings/NSGAIIElitistMultiobjectiveGA.pdf>
//!
//! If you happen to implement another kind of genetic algorithm using this
//! framework, please contribute. The more options one has - the better.
//!
//! # Operators and Executors
//!
//! Abstractions, that perform each step, are called **operators**. For each
//! **operator**, this crate provides an **executor** - another abstraction,
//! that controls application of each operator to solutions and scores.
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
//! [`Mutation`] to each solution, just like [`Mutator`] would.
//!
//! This crate does not provide the common crossover or mutation functions you'd
//! expect to see in a usual GA focused crate. The reason for this is that the
//! crate was developed to mainly operate not on numbers, but on structs or
//! *sets* of objects of unknown type. The crate does, however, provide you with
//! a few [Selectors](crate::selector#structs) and one commonly used
//! [`GenerationTerminator`](crate::terminator::GenerationTerminator) for a good
//! measure.
//!
//! # Closures
//!
//! Each **operator** trait is implemented by respective closures. A [`Test`]
//! takes a solution of type `S` and returns an array of `f32` values - one
//! value per objective. Thus, instead of implementing [`Test`] you could use
//! a closure of type `Fn(&S) -> [f32; N]`. Consult operators' documentation
//! to see what closures implement these traits.
//!
//! Note, however, that this highly generic implementation leads to unreadable
//! compile time error messages that appear not during closure definition, but
//! during creation of an optimizer. If you are struggling with a closure, may
//! be you should implement a trait directly instead. These implementations are
//! resolved during compilation so neither approach is less performant.
//!
//! # Parallelization
//!
//! The **operators** from the 3rd column of the table above can be easily
//! parallelized by calling [`par_each()`] or [`par_batch()`] methods on them
//! (the latter isn't implemented for [`Recombination`]). This cheap conversion
//! only wraps a closure into a struct, tagging it so an **executor** will apply
//! such operator in parallel for each solution/score or to their batches of
//! equal size. And you can call these methods on closures too:
//! ```
//! # use moga::operator::*;
//! let test = |f: &f32| [f + 1.0, f * 2.0];
//! let par_test = test.par_batch();
//! ```
//!
//! For simple operators the overhead introduced by parallelization usually
//! only decreases performance, but when you need it, *you need it*. Benchmark,
//! if in doubt.
//!
//! # Example
//!
//! Below lies a solution for the textbook "Binh and Korn function optimization"
//! problem.
//!
//! ```
//! // TODO: copy example from nsga2
//! ```
//!
//! You can find more examples in the *examples* folder. You can also write your
//! own and contribute it to the crate. I'd be very grateful! Here is a list of
//! functions that one day I hope to cover:
//! <https://en.wikipedia.org/wiki/Test_functions_for_optimization>
//!
//! # Use cases
//!
//! *This crate was designed to solve a very specific problem which, in case of*
//! *success, will certanly appear in this list. However, if this crate happens*
//! *to be of use to you, please contact me and I will put your repo on the list.*
//!
//! [`Optimizer`]: crate::optimizer::Optimizer
//! [`NSGA-II`]: crate::optimizer::nsga::Nsga2
//! [`Tester`]: crate::tester::Tester
//! [`Test`]: crate::tester::Test
//! [`Selector`]: crate::selector::Selector
//! [`Selection`]: crate::selector::Selection
//! [`Recombinator`]: crate::recombinator::Recombinator
//! [`Recombination`]: crate::recombinator::Recombination
//! [`Mutator`]: crate::mutator::Mutator
//! [`Mutation`]: crate::mutator::Mutation
//! [`Terminator`]: crate::terminator::Terminator
//! [`Termination`]: crate::terminator::Termination
//! [`par_each()`]: crate::operator::ParEach::par_each
//! [`par_batch()`]: crate::operator::ParBatch::par_batch

#![warn(missing_docs)]
mod execution;
pub mod mutator;
pub mod operator;
pub mod optimizer;
pub mod recombinator;
pub mod score;
pub mod selector;
pub mod terminator;
pub mod tester;
