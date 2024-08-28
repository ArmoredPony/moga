use std::cmp::Ordering;

/// Describes pareto dominance for arrays of floats.
pub(crate) trait ParetoDominance {
  /// Returns `Less` if `self` dominates `other`, `Greater` if `other`
  /// dominates `Self`, otherwise `Equal`. `self` dominates `other` if all
  /// `self` values are closer to zero than respective `other` values.
  fn dominance(&self, other: &Self) -> Ordering;
}

impl ParetoDominance for [f32] {
  fn dominance(&self, other: &Self) -> Ordering {
    let mut ord = Ordering::Equal;
    for (a, b) in self.iter().zip(other) {
      match (ord, a.abs().partial_cmp(&b.abs()).expect("NaN encountered")) {
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
    assert_eq!(
      [-2.0, 2.0, -3.0].dominance(&[-1.0, 1.0, 2.0]),
      Ordering::Greater
    );

    assert_eq!([1.0, 2.0, 3.0].dominance(&[10.0, 2.0, 3.0]), Ordering::Less);
    assert_eq!([1.0, 2.0, 3.0].dominance(&[1.0, 20.0, 3.0]), Ordering::Less);
    assert_eq!([1.0, 2.0, 3.0].dominance(&[1.0, 2.0, 30.0]), Ordering::Less);
    assert_eq!(
      [-1.0, 2.0, -3.0].dominance(&[2.0, 2.0, 4.0]),
      Ordering::Less
    );

    assert_eq!([1.0; 0].dominance(&[0.0; 0]), Ordering::Equal);
  }
}
