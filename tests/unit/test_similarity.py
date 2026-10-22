import math

import pytest

from app.domain.similarity import cosine_similarity


def test_identical_vectors_score_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_opposite_vectors_score_minus_one() -> None:
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_perpendicular_vectors_score_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 5.0]) == pytest.approx(0.0)


def test_length_does_not_matter_only_direction() -> None:
    assert cosine_similarity([1.0, 1.0], [10.0, 10.0]) == pytest.approx(1.0)


def test_known_angle() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(math.sqrt(0.5))


def test_result_stays_within_bounds_despite_rounding() -> None:
    vector = [0.1] * 1024
    assert -1.0 <= cosine_similarity(vector, vector) <= 1.0


def test_different_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        cosine_similarity([1.0, 2.0], [1.0])


def test_zero_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 2.0])
