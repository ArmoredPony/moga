use std::marker::PhantomData;

pub enum TestOperatorTag {}
pub enum SelectionOperatorTag {}
pub enum CreationOperatorTag {}
pub enum MutationOperatorTag {}
pub enum TerminationOperatorTag {}

/// A wrapper around an operator that marks it to
/// be executed in parallel for **each** solution by executor.
pub struct ParEach<OperatorTag, S, O> {
  operator: O,
  _solution: PhantomData<S>,
  _operator_tag: PhantomData<OperatorTag>,
}

impl<OperatorTag, S, O> ParEach<OperatorTag, S, O> {
  pub fn operator(&self) -> &O {
    &self.operator
  }
}

/// A wrapper around an operator that marks it to
/// be executed in parallel for each **batch** of solutions by executor.
pub struct ParBatch<OperatorTag, S, O> {
  operator: O,
  _solution: PhantomData<S>,
  _operator_tag: PhantomData<OperatorTag>,
}

impl<OperatorTag, S, O> ParBatch<OperatorTag, S, O> {
  pub fn operator(&self) -> &O {
    &self.operator
  }
}

pub trait IntoPar<OperatorTag, S, const N: usize> {
  /// Creates a wrapper around given operator that marks it to
  /// be executed in parallel for **each** solution.
  ///
  /// **Parallelization is implemented with [rayon]. As a result, for simple
  /// functions, parallelization may only decrease performance because of
  /// additional overhead introduced. Benchmark if in doubt.**
  fn par_each(self) -> ParEach<OperatorTag, S, Self>
  where
    Self: Sized,
  {
    ParEach {
      operator: self,
      _solution: PhantomData,
      _operator_tag: PhantomData,
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
  fn par_batch(self) -> ParBatch<OperatorTag, S, Self>
  where
    Self: Sized,
  {
    ParBatch {
      operator: self,
      _solution: PhantomData,
      _operator_tag: PhantomData,
    }
  }
}
