pub mod nsga;

pub use nsga::*;

use crate::score::Scores;

/// Represents an abstract optimizer.
pub trait Optimizer<Sol, const OBJECTIVE_CNT: usize>: Sized {
  /// Runs `Optimizer` until the termination condition is met, then returns
  /// last found solutions.
  fn run(mut self) -> Vec<Sol> {
    let scores = self.evaluate(self.peek_solutions());
    self.set_scores(scores);

    while !self.terminate() {
      let mut solutions = self.take_solutions();
      let mut scores = self.take_scores();

      let selected_solutions = self.select(&solutions, &scores);
      let mut created_solutions = self.create(&selected_solutions);
      created_solutions.iter_mut().for_each(|s| self.mutate(s));
      let mut created_scores: Vec<_> = self.evaluate(&created_solutions);

      solutions.append(&mut created_solutions);
      scores.append(&mut created_scores);

      let (solutions, scores) = self.truncate(solutions, scores);
      self.set_solutions(solutions);
      self.set_scores(scores);
    }

    self.take_solutions()
  }

  // TODO: add docs

  fn peek_solutions(&self) -> &[Sol];

  fn peek_scores(&self) -> &[Scores<OBJECTIVE_CNT>];

  fn take_solutions(&mut self) -> Vec<Sol>;

  fn take_scores(&mut self) -> Vec<Scores<OBJECTIVE_CNT>>;

  fn set_solutions(&mut self, solutions: Vec<Sol>);

  fn set_scores(&mut self, scores: Vec<Scores<OBJECTIVE_CNT>>);

  fn evaluate(&self, solutions: &[Sol]) -> Vec<Scores<OBJECTIVE_CNT>>;

  fn select<'a>(
    &mut self,
    solutions: &'a [Sol],
    scores: &[Scores<OBJECTIVE_CNT>],
  ) -> Vec<&'a Sol>;

  fn create(&self, solutions: &[&Sol]) -> Vec<Sol>;

  fn mutate(&self, solution: &mut Sol);

  fn truncate(
    &self,
    solutions: Vec<Sol>,
    scores: Vec<Scores<OBJECTIVE_CNT>>,
  ) -> (Vec<Sol>, Vec<Scores<OBJECTIVE_CNT>>);

  fn terminate(&mut self) -> bool;
}
