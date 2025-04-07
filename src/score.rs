//! Type aliases for a more convenient representation of fitness scores used
//! throughout the library.

use std::cmp::Ordering;

/// An alias for a fitness score.
///
/// Whenever the framework expects these scores, it tries to minimize them.
/// If you want to maximize them instead, then multiply by `-1`.
pub type Score = f32;

/// An alias for an array of `N` values of `Score` type.
pub type Scores<const N: usize> = [Score; N];

/// Describes pareto dominance for arrays of `Score`s.
pub(crate) trait ParetoDominance {
  /// Returns `Less` if `self` dominates `other`, `Greater` if `other`
  /// dominates `Self`, otherwise `Equal`. `self` dominates `other` if all
  /// `self` values are closer to zero than respective `other` values.
  fn dominance(&self, other: &Self) -> Ordering;
}

impl ParetoDominance for [Score] {
  fn dominance(&self, other: &Self) -> Ordering {
    let mut ord = Ordering::Equal;
    for (a, b) in self.iter().zip(other) {
      match (ord, a.partial_cmp(b).expect("NaN encountered")) {
        (Ordering::Equal, next_ord) => ord = next_ord,
        (Ordering::Greater, Ordering::Less)
        | (Ordering::Less, Ordering::Greater) => return Ordering::Equal,
        _ => {}
      }
    }
    ord
  }
}

#[cfg(test)]
mod tests {
  use std::cmp::Ordering;

  use super::*;

  #[test]
  fn test_pareto_dominance() {
    assert_eq!([1.0, 2.0, 3.0].dominance(&[1.0, 2.0, 3.0]), Ordering::Equal);
    assert_eq!(
      [-1.0, 2.0, -3.0].dominance(&[-1.0, 2.0, -3.0]),
      Ordering::Equal
    );
    assert_eq!(
      [-2.0, 1.0, 3.0].dominance(&[2.0, -1.0, -3.0]),
      Ordering::Equal
    );

    assert_eq!([1.0, 2.0, 3.0].dominance(&[3.0, 2.0, 1.0]), Ordering::Equal);
    assert_eq!(
      [1.0, -2.0, 3.0].dominance(&[-3.0, 2.0, 1.0]),
      Ordering::Equal
    );

    assert_eq!(
      [10.0, 2.0, 3.0].dominance(&[1.0, 2.0, 3.0]),
      Ordering::Greater
    );
    assert_eq!(
      [1.0, 20.0, 3.0].dominance(&[1.0, 2.0, 3.0]),
      Ordering::Greater
    );
    assert_eq!(
      [1.0, 2.0, 30.0].dominance(&[1.0, 2.0, 3.0]),
      Ordering::Greater
    );

    assert_eq!([1.0, 2.0, 3.0].dominance(&[10.0, 2.0, 3.0]), Ordering::Less);
    assert_eq!([1.0, 2.0, 3.0].dominance(&[1.0, 20.0, 3.0]), Ordering::Less);
    assert_eq!([1.0, 2.0, 3.0].dominance(&[1.0, 2.0, 30.0]), Ordering::Less);

    assert_eq!([1.0; 0].dominance(&[0.0; 0]), Ordering::Equal);
  }
}
