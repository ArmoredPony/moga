use std::{marker::PhantomData, ops::Deref};

/// Sequential execution strategy marker, i.e. no parallelization involved.
pub enum SequentialExecution {}

/// Parallel execution strategy marker, parallelizes objective testing for
/// **each** solution.
pub enum ParallelEachExecution {}

/// Parallel execution strategy marker, parallelizes objective testing for a
/// **batch** of solutions. The crate tries to split the work equally for each
/// available thread.
pub enum ParallelBatchExecution {}

/// Custom execution strategy marker.
pub enum CustomExecution {}

/// A wrapper around an operator that marks it to
/// be executed in parallel for **each** solution by executor.
pub struct ParEach<S, O> {
  operator: O,
  _solution: PhantomData<S>,
}

impl<S, O> Deref for ParEach<S, O> {
  type Target = O;

  fn deref(&self) -> &Self::Target {
    &self.operator
  }
}

/// A wrapper around an operator that marks it to
/// be executed in parallel for each **batch** of solutions by executor.
pub struct ParBatch<S, O> {
  operator: O,
  _solution: PhantomData<S>,
}

impl<S, O> Deref for ParBatch<S, O> {
  type Target = O;

  fn deref(&self) -> &Self::Target {
    &self.operator
  }
}
pub trait IntoPar<S, const N: usize> {
  /// Creates a wrapper around given operator that marks it to
  /// be executed in parallel for **each** solution.
  ///
  /// **Parallelization is implemented with [rayon]. As a result, for simple
  /// functions, parallelization may only decrease performance because of
  /// additional overhead introduced. Benchmark if in doubt.**
  fn par_each(self) -> ParEach<S, Self>
  where
    Self: Sized,
  {
    ParEach {
      operator: self,
      _solution: PhantomData,
    }
  }

  /// Creates a wrapper around given operator that marks it to
  /// be executed in parallel for each **batch** of solutions.
  /// The crate calculates the size of the batch in such a way as to evenly
  /// distribute the calculations across all available threads.
  ///
  /// **Parallelization is implemented with [rayon]. As a result, for simple
  /// functions, parallelization may only decrease performance because of
  /// additional overhead introduced. Benchmark if in doubt.**
  fn par_batch(self) -> ParBatch<S, Self>
  where
    Self: Sized,
  {
    ParBatch {
      operator: self,
      _solution: PhantomData,
    }
  }
}
