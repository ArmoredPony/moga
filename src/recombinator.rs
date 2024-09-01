use itertools::Itertools;
use rayon::prelude::*;

use crate::{execution::*, operator::*};

/// Creates new solutions by all possible unique parents' permutations of
/// length `P`. Each permutation produces `O` offsprings.
pub trait RecombinationOperator<S, const P: usize, const O: usize> {
  /// Takes references to a selected parents's combination of length `P`
  /// and returns `O` created offsprings.   
  fn recombine(&self, parents: [&S; P]) -> [S; O];
}

impl<S, const P: usize, const O: usize, F> RecombinationOperator<S, P, O> for F
where
  F: Fn([&S; P]) -> [S; O],
{
  fn recombine(&self, parents: [&S; P]) -> [S; O] {
    self(parents)
  }
}

// impl<S, F> RecombinationOperator<S, 1, 1> for F
// where
//   F: Fn(&S) -> S,
// {
//   fn recombine(&self, solutions: [&S; 1]) -> [S; 1] {
//     [self(solutions[0])]
//   }
// }

// impl<S, F> RecombinationOperator<S, 2, 1> for F
// where
//   F: Fn(&S, &S) -> S,
// {
//   fn recombine(&self, solutions: &[&S]) -> Vec<S> {
//     if solutions.is_empty() {
//       return vec![];
//     }
//     (0..solutions.len() - 1)
//       .flat_map(|i| {
//         (i + 1..solutions.len()).map(move |j| (&solutions[i], &solutions[j]))
//       })
//       .map(|(a, b)| self(a, b))
//       .collect()
//   }
// }

// // tuple-to-array conversion can be implemented with a macro
// impl<S, F> RecombinationOperator<S, 2, 2> for F
// where
//   F: Fn(&S, &S) -> (S, S),
// {
//   fn recombine(&self, solutions: &[&S]) -> Vec<S> {
//     if solutions.is_empty() {
//       return vec![];
//     }
//     (0..solutions.len() - 1)
//       .flat_map(|i| {
//         (i + 1..solutions.len()).map(move |j| (&solutions[i], &solutions[j]))
//       })
//       .flat_map(|(a, b)| <[S; 2]>::from(self(a, b)))
//       .collect()
//   }
// }

impl<S, R, const N: usize> ParEach<RecombinationOperatorTag, S, N> for R where
  R: RecombinationOperator<S, N, N> // does this even work?
{
}

/// Creates offsprings by recombining previously selected parents.
trait Recombinator<S> {
  /// Recombines given parents, returning a vector of newly created offsprings.
  fn recombine(&self, parents: &[&S]) -> Vec<S>;
}

impl<S, F> Recombinator<S> for F
where
  F: Fn(&[&S]) -> Vec<S>,
{
  fn recombine(&self, parents: &[&S]) -> Vec<S> {
    self(parents)
  }
}

// TODO: add docs
// TODO: make private
pub trait RecombinationExecutor<
  S,
  const P: usize,
  const O: usize,
  ExecutionStrategy,
>
{
  fn execute_recombination(&self, parents: &[&S]) -> Vec<S>;
}

impl<S, const P: usize, const O: usize, R>
  RecombinationExecutor<S, P, O, CustomExecutionStrategy> for R
where
  R: Recombinator<S>,
{
  fn execute_recombination(&self, parents: &[&S]) -> Vec<S> {
    self.recombine(parents)
  }
}

impl<S, const P: usize, const O: usize, R>
  RecombinationExecutor<S, P, O, SequentialExecutionStrategy> for R
where
  R: RecombinationOperator<S, P, O>,
{
  fn execute_recombination(&self, parents: &[&S]) -> Vec<S> {
    parents
      .iter()
      .copied()
      .combinations(P)
      .flat_map(|c| {
        self.recombine(
          c.try_into()
            .unwrap_or_else(|_| panic!("this conversion should succeed")),
        )
      })
      .collect()
  }
}

impl<S, const P: usize, const O: usize, R>
  RecombinationExecutor<S, P, O, ParallelEachExecutionStrategy>
  for ParEachOperator<RecombinationOperatorTag, S, R>
where
  S: Sync + Send,
  R: RecombinationOperator<S, P, O> + Sync + Send,
{
  fn execute_recombination(&self, parents: &[&S]) -> Vec<S> {
    parents
      .iter()
      .copied()
      .combinations(P)
      .par_bridge()
      .flat_map_iter(|c| {
        self.operator().recombine(
          c.try_into()
            .unwrap_or_else(|_| panic!("this conversion should succeed")),
        )
      })
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  type Solution = f64;

  const fn as_crossover<
    const P: usize,
    const O: usize,
    C: RecombinationOperator<Solution, P, O>,
  >(
    _: &C,
  ) {
  }

  // #[test]
  // fn test_crossover_1_to_1() {
  //   let c = |s: &Solution| s.to_owned() + 1.0;
  //   as_crossover(&c);

  //   let parents: Vec<_> = (0..100).map(Solution::from).collect();
  //   let offsprings = c.recombine(&parents.iter().collect::<Vec<_>>());

  //   assert_eq!(parents.len(), offsprings.len());
  //   assert_eq!(c.recombine(&[]), &[]);
  // }

  // #[test]
  // fn test_crossover_2_to_1() {
  //   let c = |a: &Solution, b: &Solution| a + b;
  //   as_crossover(&c);

  //   let parents: Vec<_> = (0..100).map(Solution::from).collect();
  //   let offsprings = c.recombine(&parents.iter().collect::<Vec<_>>());

  //   assert_eq!(offsprings.len(), (0..parents.len()).sum());

  //   assert_eq!(c.recombine(&[]), &[]);
  //   assert_eq!(c.recombine(&[&1.0]), &[]);
  //   assert_eq!(c.recombine(&[&1.0, &2.0]), &[3.0]);
  // }

  // #[test]
  // fn test_crossover_2_to_2() {
  //   let c = |a: &Solution, b: &Solution| (a + b, a - b);
  //   as_crossover(&c);

  //   let parents: Vec<_> = (0..100).map(Solution::from).collect();
  //   let offsprings = c.recombine(&parents.iter().collect::<Vec<_>>());

  //   assert_eq!(offsprings.len(), (0..parents.len()).sum::<usize>() * 2);

  //   assert_eq!(c.recombine(&[]), &[]);
  //   assert_eq!(c.recombine(&[&1.0]), &[]);
  //   assert_eq!(c.recombine(&[&1.0, &2.0]), &[3.0, -1.0]);
  // }

  // #[test]
  // fn test_crossover_n_to_m() {
  //   let c = |solutions: &[&Solution]| {
  //     solutions
  //       .chunks_exact(2)
  //       .map(|p| Solution::max(*p[0], *p[1]))
  //       .collect::<Vec<_>>()
  //   };
  //   as_crossover(&c);

  //   let parents: Vec<_> = (0..100).map(Solution::from).collect();
  //   let offsprings = c.recombine(&parents.iter().collect::<Vec<_>>());

  //   assert_eq!(offsprings.len(), parents.len() / 2);

  //   assert_eq!(c.recombine(&[]), &[]);
  //   assert_eq!(c.recombine(&[&1.0]), &[]);
  //   assert_eq!(c.recombine(&[&1.0, &2.0]), &[2.0]);
  //   assert_eq!(c.recombine(&[&1.0, &2.0, &3.0]), &[2.0]);
  //   assert_eq!(c.recombine(&[&1.0, &2.0, &3.0, &4.0]), &[2.0, 4.0]);
  // }
}
