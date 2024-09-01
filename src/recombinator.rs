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

macro_rules! recombination_operator_fn_impl {
  (
    ($par:ident $(, $pars:ident)*),
    ($par_nam:ident $(, $par_nams:ident)*),
    $par_cnt:expr,
    ($off:ident $(, $offs:ident)*),
    ($off_nam:ident $(, $off_nams:ident)*),
    $off_cnt:expr
  ) => {
    #[allow(unused_parens)]
    impl<$par, F> RecombinationOperator<$par, $par_cnt, $off_cnt> for F
    where
      F: Fn( &$par $(, &$pars)* ) -> ( $off $(, $offs)* ),
    {
      fn recombine(&self, parents: [&$par; $par_cnt]) -> [$off; $off_cnt] {
        let ( $par_nam, $( $par_nams, )* ) = parents.into();
        let ( $off_nam $(, $off_nams )* ) = self($par_nam $(, $par_nams)*);
        [ $off_nam, $( $off_nams, )* ]
      }
    }
  };
}

recombination_operator_fn_impl! {(S), (a), 1, (S), (m), 1} // disgusting stuff
recombination_operator_fn_impl! {(S), (a), 1, (S, S), (m, n), 2}
recombination_operator_fn_impl! {(S), (a), 1, (S, S, S), (m, n, o), 3}
recombination_operator_fn_impl! {(S), (a), 1, (S, S, S, S), (m, n, o, p), 4}
recombination_operator_fn_impl! {(S, S), (a, b), 2, (S), (m), 1}
recombination_operator_fn_impl! {(S, S), (a, b), 2, (S, S), (m, n), 2}
recombination_operator_fn_impl! {(S, S), (a, b), 2, (S, S, S), (m, n, o), 3}
recombination_operator_fn_impl! {(S, S), (a, b), 2, (S, S, S, S), (m, n, o, p), 4}
recombination_operator_fn_impl! {(S, S, S), (a, b, c), 3, (S), (m), 1}
recombination_operator_fn_impl! {(S, S, S), (a, b, c), 3, (S, S), (m, n), 2}
recombination_operator_fn_impl! {(S, S, S), (a, b, c), 3, (S, S, S), (m, n, o), 3}
recombination_operator_fn_impl! {(S, S, S), (a, b, c), 3, (S, S, S, S), (m, n, o, p), 4}
recombination_operator_fn_impl! {(S, S, S, S), (a, b, c, d), 4, (S), (m), 1}
recombination_operator_fn_impl! {(S, S, S, S), (a, b, c, d), 4, (S, S), (m, n), 2}
recombination_operator_fn_impl! {(S, S, S, S), (a, b, c, d), 4, (S, S, S), (m, n, o), 3}
recombination_operator_fn_impl! {(S, S, S, S), (a, b, c, d), 4, (S, S, S, S), (m, n, o, p), 4}

impl<S, R, const P: usize, const O: usize>
  ParEach<RecombinationOperatorTag, S, P, O> for R
where
  R: RecombinationOperator<S, P, O>,
{
}

/// Creates offsprings by recombining previously selected parents.
pub trait Recombinator<S> {
  /// Recombines given parents, returning a vector of newly created offsprings.
  fn recombine(&self, parents: Vec<&S>) -> Vec<S>;
}

impl<S, F> Recombinator<S> for F
where
  F: Fn(Vec<&S>) -> Vec<S>,
{
  fn recombine(&self, parents: Vec<&S>) -> Vec<S> {
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
  fn execute_recombination(&self, parents: Vec<&S>) -> Vec<S>;
}

impl<S, R>
  RecombinationExecutor<
    S,
    { usize::MAX },
    { usize::MAX },
    CustomExecutionStrategy,
  > for R
where
  R: Recombinator<S>,
{
  fn execute_recombination(&self, parents: Vec<&S>) -> Vec<S> {
    self.recombine(parents)
  }
}

impl<S, const P: usize, const O: usize, R>
  RecombinationExecutor<S, P, O, SequentialExecutionStrategy> for R
where
  R: RecombinationOperator<S, P, O>,
{
  fn execute_recombination(&self, parents: Vec<&S>) -> Vec<S> {
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
  fn execute_recombination(&self, parents: Vec<&S>) -> Vec<S> {
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

  fn takes_recombinator<
    const P: usize,
    const O: usize,
    ES,
    R: RecombinationExecutor<Solution, P, O, ES>,
  >(
    r: &R,
  ) {
    r.execute_recombination(vec![]);
  }

  #[test]
  fn test_recombination_operator_from_closure_1_to_1() {
    let r = |_: &Solution| 0.0;
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_1_to_2() {
    let r = |_: &Solution| (0.0, 0.0);
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_1_to_3() {
    let r = |_: &Solution| (0.0, 0.0, 0.0);
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_1_to_4() {
    let r = |_: &Solution| (0.0, 0.0, 0.0, 0.0);
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_2_to_1() {
    let r = |_: &Solution, _: &Solution| 0.0;
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_2_to_2() {
    let r = |_: &Solution, _: &Solution| (0.0, 0.0);
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_2_to_3() {
    let r = |_: &Solution, _: &Solution| (0.0, 0.0, 0.0);
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_2_to_4() {
    let r = |_: &Solution, _: &Solution| (0.0, 0.0, 0.0, 0.0);
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_3_to_1() {
    let r = |_: &Solution, _: &Solution, _: &Solution| 0.0;
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_3_to_2() {
    let r = |_: &Solution, _: &Solution, _: &Solution| (0.0, 0.0);
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_3_to_3() {
    let r = |_: &Solution, _: &Solution, _: &Solution| (0.0, 0.0, 0.0);
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_3_to_4() {
    let r = |_: &Solution, _: &Solution, _: &Solution| (0.0, 0.0, 0.0, 0.0);
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_4_to_1() {
    let r = |_: &Solution, _: &Solution, _: &Solution, _: &Solution| 0.0;
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_4_to_2() {
    let r = |_: &Solution, _: &Solution, _: &Solution, _: &Solution| (0.0, 0.0);
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_4_to_3() {
    let r =
      |_: &Solution, _: &Solution, _: &Solution, _: &Solution| (0.0, 0.0, 0.0);
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombination_operator_from_closure_4_to_4() {
    let r = |_: &Solution, _: &Solution, _: &Solution, _: &Solution| {
      (0.0, 0.0, 0.0, 0.0)
    };
    takes_recombinator(&r);
    takes_recombinator(&r.par_each());
  }

  #[test]
  fn test_recombinator_from_closure() {
    let r = |_: Vec<&Solution>| vec![0.0, 1.0, 2.0, 3.0, 4.0];
    takes_recombinator(&r);
  }

  #[test]
  fn test_custom_recombination_operator() {
    struct CustomRecombinationOperator {}
    impl<S: Copy> RecombinationOperator<S, 1, 3> for CustomRecombinationOperator {
      fn recombine(&self, parents: [&S; 1]) -> [S; 3] {
        [parents[0].to_owned(); 3]
      }
    }

    let r = CustomRecombinationOperator {};
    takes_recombinator(&r)
  }

  #[test]
  fn test_custom_recombinator() {
    struct CustomRecombinator {}
    impl<S: Copy> Recombinator<S> for CustomRecombinator {
      fn recombine(&self, parents: Vec<&S>) -> Vec<S> {
        parents.into_iter().copied().collect()
      }
    }

    let r = CustomRecombinator {};
    takes_recombinator(&r);
  }
}
