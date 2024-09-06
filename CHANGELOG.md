# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

### Removed

- Removed `BestSelector` and `TournamentSelector`. These selectors select
  solutions with minimal sum of their objective scores, favoring solutions, that
  do better for objectives with smaller maximum possible value. This approach
  harms diversity of population.

## [0.1.0]

### Added

- **MOGA** is released on [crates.io](https://crates.io/crates/moga). Hello,
  MOGA!

[unreleased]: https://github.com/ArmoredPony/moga/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ArmoredPony/moga/releases/tag/v0.1.0