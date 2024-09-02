pub(crate) mod strategy {
  /// Sequential execution strategy marker, i.e. no parallelization involved.
  pub enum SequentialExecutionStrategy {}

  /// Parallel execution strategy marker, parallelizes objective testing for
  /// **each** solution.
  pub enum ParallelEachExecutionStrategy {}

  /// Parallel execution strategy marker, parallelizes objective testing for a
  /// **batch** of solutions. The crate tries to split the work equally for each
  /// available thread.
  pub enum ParallelBatchExecutionStrategy {}

  /// Custom execution strategy marker.
  pub enum CustomExecutionStrategy {}
}
