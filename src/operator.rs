//! Common GA operators' utilities.

use std::marker::PhantomData;

pub(crate) mod tag {
  #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
  pub enum TestOperatorTag {}
  #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
  pub enum SelectionOperatorTag {}
  #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
  pub enum RecombinationOperatorTag {}
  #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
  pub enum MutationOperatorTag {}
  #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
  pub enum TerminationOperatorTag {}
}

/// A wrapper around an operator that marks it to
/// be executed in parallel for **each** solution by the executor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ParEachOperator<OperatorTag, S, O> {
  operator: O,
  _solution: PhantomData<S>,
  _operator_tag: PhantomData<OperatorTag>,
}

impl<OperatorTag, S, O> ParEachOperator<OperatorTag, S, O> {
  pub(crate) fn operator(&self) -> &O {
    &self.operator
  }
}

/// Used to tag an operator as parallelized for **each** solution.
///
/// This conversion is cheap and doesn't change operator's behavior. However,
/// executors treat tagged operators differently.
pub trait ParEach<OperatorTag, S, const N: usize, const M: usize> {
  /// Creates a wrapper around the given operator that marks it to
  /// be executed in parallel for **each** solution.
  ///
  /// **Parallelization is implemented with [rayon]. As a result, for simple
  /// functions, parallelization may only decrease performance because of
  /// additional overhead introduced. Benchmark if in doubt.**
  fn par_each(self) -> ParEachOperator<OperatorTag, S, Self>
  where
    Self: Sized,
  {
    ParEachOperator {
      operator: self,
      _solution: PhantomData,
      _operator_tag: PhantomData,
    }
  }
}

/// A wrapper around an operator that marks it to
/// be executed in parallel for each **batch** of solutions by the executor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ParBatchOperator<OperatorTag, S, O> {
  operator: O,
  _solution: PhantomData<S>,
  _operator_tag: PhantomData<OperatorTag>,
}

impl<OperatorTag, S, O> ParBatchOperator<OperatorTag, S, O> {
  pub(crate) fn operator(&self) -> &O {
    &self.operator
  }
}

/// Used to tag an operator as parallelized for each **batch** of solutions.
///
/// This conversion is cheap and doesn't change operator's behavior. However,
/// executors treat tagged operators differently.
pub trait ParBatch<OperatorTag, S, const N: usize> {
  /// Creates a wrapper around the given operator that marks it to
  /// be executed in parallel for each **batch** of solutions.
  /// The crate calculates the size of the batch in such a way as to evenly
  /// distribute the calculations across all available threads. This is usually
  /// faster than parallelization for each individual solution.
  ///
  /// **Parallelization is implemented with [rayon]. As a result, for simple
  /// functions, parallelization may only decrease performance because of
  /// additional overhead introduced. Benchmark if in doubt.**
  fn par_batch(self) -> ParBatchOperator<OperatorTag, S, Self>
  where
    Self: Sized,
  {
    ParBatchOperator {
      operator: self,
      _solution: PhantomData,
      _operator_tag: PhantomData,
    }
  }
}
