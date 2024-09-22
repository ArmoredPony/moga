# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

### Added

- Added implementation of [SPEA-II] - improved version of the Strength Pareto
  Evolutionary Algorithm.

- Added plots of objective functions' values for each example along with scripts
  that create plots from the examples. The plots can be found in the *examples*
  folder in the root of the project.

- Added `RouletteSelector`, `TournamentSelectorWithReplacement` and
  `TournamentSelectorWithoutReplacement` selectors. These selectors select
  solutions based on their number of dominations over each other.

### Changed

- `ParEach` and `ParBatch` operator wrappers now put constraints on their
  operators and the solution type. This helps to catch parallelization errors
  sooner.

- Miscellaneous documentation updates.

- Miscellaneous examples updates.

### Removed

- Removed all re-exports from `lib.rs`.

- Removed `BestSelector` and `TournamentSelector` original implementations.
  These selectors used to select solutions with minimal sum of their objective
  scores. This approach harmed diversity of population, favoring solutions, that
  leaned towards objectives with the smaller upper bound.

## [0.1.0]

### Added

- **MOGA** is released on [crates.io](https://crates.io/crates/moga). Hello,
  MOGA!

[unreleased]: https://github.com/ArmoredPony/moga/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ArmoredPony/moga/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ArmoredPony/moga/releases/tag/v0.1.0

[SPEA-II]: https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/145755/eth-24689-01.pdf