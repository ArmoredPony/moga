pub mod nsga;

pub use nsga::*;

use crate::score::Scores;

/// Represents an abstract optimizer.
pub trait Optimizer<Solution, const OBJECTIVE_NUM: usize>: Sized {
  /// Runs `Optimizer` until the termination condition is met, then returns
  /// last found solutions.
  fn optimize(mut self) -> Vec<Solution> {
    let scores = self.test(self.peek_solutions());
    self.set_scores(scores);

    while !self.terminate() {
      let mut solutions = self.take_solutions();
      let mut scores = self.take_scores();

      let selected_solutions = self.select(&solutions, &scores);
      let mut created_solutions = self.create(&selected_solutions);
      self.mutate(&mut created_solutions);
      let mut created_scores = self.test(&created_solutions);

      solutions.append(&mut created_solutions);
      scores.append(&mut created_scores);

      let (solutions, scores) = self.truncate(solutions, scores);
      self.set_solutions(solutions);
      self.set_scores(scores);
    }

    self.take_solutions()
  }

  /// Returns a slice of solutions.
  fn peek_solutions(&self) -> &[Solution];

  /// Returns a slice of solutions' fitness scores.
  fn peek_scores(&self) -> &[Scores<OBJECTIVE_NUM>];

  /// Moves soulutions out from `Optimizer`.
  fn take_solutions(&mut self) -> Vec<Solution>;

  /// Moves soulutions' scores out from `Optimizer`.
  fn take_scores(&mut self) -> Vec<Scores<OBJECTIVE_NUM>>;

  /// Sets solutions in `Optimizer`.
  fn set_solutions(&mut self, solutions: Vec<Solution>);

  /// Sets scores in `Optimizer`.
  fn set_scores(&mut self, scores: Vec<Scores<OBJECTIVE_NUM>>);

  /// Tests each solution and returns a vector of their fitness scores.
  fn test(&self, solutions: &[Solution]) -> Vec<Scores<OBJECTIVE_NUM>>;

  /// Selects solutions suitable for creation of new solutions.
  fn select<'a>(
    &mut self,
    solutions: &'a [Solution],
    scores: &[Scores<OBJECTIVE_NUM>],
  ) -> Vec<&'a Solution>;

  /// Creates new solutions from previously selected ones.
  fn create(&self, solutions: &[&Solution]) -> Vec<Solution>;

  /// Mutates solutions.
  fn mutate(&self, solution: &mut [Solution]);

  /// Truncates excessive solutions. Implementation of truncation operator is
  /// specific for each `Optimizer` implementation.
  fn truncate(
    &self,
    solutions: Vec<Solution>,
    scores: Vec<Scores<OBJECTIVE_NUM>>,
  ) -> (Vec<Solution>, Vec<Scores<OBJECTIVE_NUM>>);

  /// Terminates algorithm.
  fn terminate(&mut self) -> bool;
}
