mod genetic_algorithm;
pub mod nsga;

use genetic_algorithm::GeneticAlgorithm;

/// Represents an abstract optimizer.
pub trait Optimizer<Solution, const OBJECTIVE_NUM: usize>:
  GeneticAlgorithm<Solution, OBJECTIVE_NUM> + Sized
{
  /// Runs `Optimizer` until the termination condition is met, then returns
  /// the last found population.
  fn optimize(mut self) -> Vec<Solution> {
    let scores = self.test(self.peek_population());
    self.set_scores(scores);

    while !self.terminate() {
      let mut population = self.take_population();
      let mut scores = self.take_scores();

      let selected_population = self.select(&population, &scores);
      let mut created_population = self.create(selected_population);
      self.mutate(&mut created_population);
      let mut created_scores = self.test(&created_population);

      population.append(&mut created_population);
      scores.append(&mut created_scores);

      let (population, scores) = self.truncate(population, scores);
      self.set_population(population);
      self.set_scores(scores);
    }

    self.take_population()
  }
}

impl<S, const N: usize, G: GeneticAlgorithm<S, N>> Optimizer<S, N> for G {}
