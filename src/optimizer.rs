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
      created_solutions.iter_mut().for_each(|s| self.mutate(s));
      let mut created_scores: Vec<_> = self.test(&created_solutions);

      solutions.append(&mut created_solutions);
      scores.append(&mut created_scores);

      let (solutions, scores) = self.truncate(solutions, scores);
      self.set_solutions(solutions);
      self.set_scores(scores);
    }

    self.take_solutions()
  }

  // TODO: add docs

  fn peek_solutions(&self) -> &[Solution];

  fn peek_scores(&self) -> &[Scores<OBJECTIVE_NUM>];

  fn take_solutions(&mut self) -> Vec<Solution>;

  fn take_scores(&mut self) -> Vec<Scores<OBJECTIVE_NUM>>;

  fn set_solutions(&mut self, solutions: Vec<Solution>);

  fn set_scores(&mut self, scores: Vec<Scores<OBJECTIVE_NUM>>);

  fn test(&self, solutions: &[Solution]) -> Vec<Scores<OBJECTIVE_NUM>>;

  fn select<'a>(
    &mut self,
    solutions: &'a [Solution],
    scores: &[Scores<OBJECTIVE_NUM>],
  ) -> Vec<&'a Solution>;

  fn create(&self, solutions: &[&Solution]) -> Vec<Solution>;

  fn mutate(&self, solution: &mut Solution);

  fn truncate(
    &self,
    solutions: Vec<Solution>,
    scores: Vec<Scores<OBJECTIVE_NUM>>,
  ) -> (Vec<Solution>, Vec<Scores<OBJECTIVE_NUM>>);

  fn terminate(&mut self) -> bool;
}
