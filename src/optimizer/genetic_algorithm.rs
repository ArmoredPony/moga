use crate::score::Scores;

/// Represents an abstract genetic algorithm.
pub trait GeneticAlgorithm<Solution, const OBJECTIVE_NUM: usize> {
  /// Returns a slice of population.
  fn peek_population(&self) -> &[Solution];

  /// Returns a slice of population fitness scores.
  fn peek_scores(&self) -> &[Scores<OBJECTIVE_NUM>];

  /// Moves population out from `Optimizer`.
  fn take_population(&mut self) -> Vec<Solution>;

  /// Moves population scores out from `Optimizer`.
  fn take_scores(&mut self) -> Vec<Scores<OBJECTIVE_NUM>>;

  /// Sets population in `Optimizer`.
  fn set_population(&mut self, population: Vec<Solution>);

  /// Sets scores in `Optimizer`.
  fn set_scores(&mut self, scores: Vec<Scores<OBJECTIVE_NUM>>);

  /// Tests solutions in population and returns a vector of their
  /// fitness scores.
  fn test(&self, population: &[Solution]) -> Vec<Scores<OBJECTIVE_NUM>>;

  /// Selects solutions from population that suitable for creation of new
  /// population.
  fn select<'a>(
    &mut self,
    population: &'a [Solution],
    scores: &[Scores<OBJECTIVE_NUM>],
  ) -> Vec<&'a Solution>;

  /// Creates new population from selected solutions of previous population.
  fn create(&self, population: Vec<&Solution>) -> Vec<Solution>;

  /// Mutates population.
  fn mutate(&self, population: &mut [Solution]);

  /// Truncates excessive solutions from population.
  /// Truncation operator is specific for each `Optimizer` implementation.
  fn truncate(
    &self,
    population: Vec<Solution>,
    scores: Vec<Scores<OBJECTIVE_NUM>>,
  ) -> (Vec<Solution>, Vec<Scores<OBJECTIVE_NUM>>);

  /// Terminates algorithm.
  fn terminate(&mut self) -> bool;
}
