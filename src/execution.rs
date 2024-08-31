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
