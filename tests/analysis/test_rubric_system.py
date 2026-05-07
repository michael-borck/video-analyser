"""Tests for custom rubric evaluation system."""

import tempfile
from pathlib import Path

import pytest

from video_analyser.analysis.rubric_system import (
    Rubric,
    RubricAssessment,
    RubricBuilder,
    RubricCategory,
    RubricCriterion,
    RubricRepository,
    RubricScorer,
    ScoringScale,
)


class TestRubricCriterion:
    """Tests for RubricCriterion model."""

    def test_create_criterion(self):
        """Test creating a rubric criterion."""
        criterion = RubricCriterion(
            name="Clear Thesis",
            description="Student clearly states main argument",
            weight=2.0,
        )
        assert criterion.name == "Clear Thesis"
        assert criterion.description == "Student clearly states main argument"
        assert criterion.weight == 2.0

    def test_criterion_with_scoring_guide(self):
        """Test criterion with scoring guide."""
        criterion = RubricCriterion(
            name="Organization",
            scoring_guide="5: Excellent flow; 3: Adequate structure; 1: Disorganized",
            level_descriptions={1: "Disorganized", 3: "Adequate", 5: "Excellent"},
        )
        assert criterion.scoring_guide is not None
        assert len(criterion.level_descriptions) == 3

    def test_criterion_default_weight(self):
        """Test criterion defaults to weight of 1.0."""
        criterion = RubricCriterion(name="Test Criterion")
        assert criterion.weight == 1.0


class TestRubricCategory:
    """Tests for RubricCategory model."""

    def test_create_category(self):
        """Test creating a rubric category."""
        criterion = RubricCriterion(name="Criterion 1")
        category = RubricCategory(
            name="Content", description="Content quality", criteria=[criterion]
        )
        assert category.name == "Content"
        assert len(category.criteria) == 1

    def test_category_requires_criteria(self):
        """Test that category requires at least one criterion."""
        with pytest.raises(ValueError, match="must have at least one criterion"):
            RubricCategory(name="Content", criteria=[])

    def test_category_weight(self):
        """Test category weight."""
        criterion = RubricCriterion(name="Criterion 1")
        category = RubricCategory(name="Content", weight=1.5, criteria=[criterion])
        assert category.weight == 1.5


class TestScoringScale:
    """Tests for ScoringScale model."""

    def test_create_scoring_scale(self):
        """Test creating a scoring scale."""
        scale = ScoringScale(
            name="1-5 Stars",
            min_score=1,
            max_score=5,
            labels={1: "Poor", 5: "Excellent"},
        )
        assert scale.min_score == 1
        assert scale.max_score == 5
        assert scale.labels[1] == "Poor"

    def test_scale_max_greater_than_min(self):
        """Test that max_score > min_score."""
        with pytest.raises(ValueError, match="must be greater than min_score"):
            ScoringScale(name="Invalid", min_score=5, max_score=5)

    def test_percentage_scale(self):
        """Test percentage scoring scale."""
        scale = ScoringScale(
            name="Percentage",
            min_score=0,
            max_score=100,
            labels={0: "0%", 100: "100%"},
        )
        assert scale.max_score == 100


class TestRubric:
    """Tests for Rubric model."""

    def test_create_rubric(self):
        """Test creating a complete rubric."""
        criterion = RubricCriterion(name="Criterion 1")
        category = RubricCategory(name="Content", criteria=[criterion])
        rubric = Rubric(name="Test Rubric", categories=[category])

        assert rubric.name == "Test Rubric"
        assert len(rubric.categories) == 1
        assert not rubric.is_template

    def test_rubric_with_tags(self):
        """Test rubric with tags."""
        criterion = RubricCriterion(name="Criterion 1")
        category = RubricCategory(name="Content", criteria=[criterion])
        rubric = Rubric(
            name="Physics Rubric",
            categories=[category],
            tags=["Physics", "Presentation", "Final Project"],
        )

        assert "Physics" in rubric.tags
        assert len(rubric.tags) == 3

    def test_rubric_serialization(self):
        """Test converting rubric to/from dict."""
        criterion = RubricCriterion(name="Criterion 1")
        category = RubricCategory(name="Content", criteria=[criterion])
        original = Rubric(name="Test", categories=[category])

        # Serialize
        data = original.to_dict()
        assert data["name"] == "Test"
        assert isinstance(data["created_at"], str)

        # Deserialize
        restored = Rubric.from_dict(data)
        assert restored.name == original.name
        assert len(restored.categories) == len(original.categories)


class TestRubricBuilder:
    """Tests for RubricBuilder helper."""

    def test_builder_creates_rubric(self):
        """Test that builder creates complete rubric."""
        builder = RubricBuilder("Presentation Rubric")
        cat_builder = builder.add_category("Content", weight=0.5)
        cat_builder.add_criterion("Clear Thesis", weight=2.0)
        cat_builder.add_criterion("Supporting Evidence", weight=1.5)

        builder.add_category("Delivery", weight=0.5).add_criterion("Eye Contact")

        rubric = builder.build(tags=["Presentation"])

        assert rubric.name == "Presentation Rubric"
        assert len(rubric.categories) == 2
        assert len(rubric.categories[0].criteria) == 2
        assert "Presentation" in rubric.tags

    def test_builder_custom_scoring_scale(self):
        """Test builder with custom scoring scale."""
        builder = RubricBuilder("Test")
        builder.set_scoring_scale(0, 10, labels={0: "Fail", 5: "Pass", 10: "Excellent"})

        criterion = RubricCriterion(name="Test")
        category = RubricCategory(name="Test", criteria=[criterion])
        builder.categories.append(category)

        rubric = builder.build()

        assert rubric.scoring_scale.max_score == 10
        assert rubric.scoring_scale.labels[5] == "Pass"

    def test_builder_fluent_interface(self):
        """Test fluent interface for builder."""
        rubric = (
            RubricBuilder("Fluent Test")
            .add_category("Test")
            .add_criterion("Criterion 1")
        )
        # Just verify it returns CategoryBuilder for chaining
        assert rubric is not None


class TestRubricRepository:
    """Tests for RubricRepository storage."""

    def test_save_and_load_rubric(self):
        """Test saving and loading rubric from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = RubricRepository(Path(tmpdir))

            # Create and save rubric
            criterion = RubricCriterion(name="Test")
            category = RubricCategory(name="Category", criteria=[criterion])
            original = Rubric(name="Test Rubric", categories=[category])

            repo.save(original)

            # Load it back
            loaded = repo.load(original.id)
            assert loaded is not None
            assert loaded.name == original.name
            assert len(loaded.categories) == 1

    def test_list_rubrics(self):
        """Test listing rubrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = RubricRepository(Path(tmpdir))

            # Create and save multiple rubrics
            for i in range(3):
                criterion = RubricCriterion(name=f"Criterion {i}")
                category = RubricCategory(name="Category", criteria=[criterion])
                rubric = Rubric(
                    name=f"Rubric {i}",
                    categories=[category],
                    is_template=(i == 0),
                )
                repo.save(rubric)

            # List all
            all_rubrics = repo.list_rubrics()
            assert len(all_rubrics) == 3

            # List templates only
            templates = repo.list_rubrics(template_only=True)
            assert len(templates) == 1

    def test_delete_rubric(self):
        """Test deleting a rubric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = RubricRepository(Path(tmpdir))

            criterion = RubricCriterion(name="Test")
            category = RubricCategory(name="Category", criteria=[criterion])
            rubric = Rubric(name="To Delete", categories=[category])

            repo.save(rubric)
            assert repo.load(rubric.id) is not None

            # Delete it
            deleted = repo.delete(rubric.id)
            assert deleted is True
            assert repo.load(rubric.id) is None

    def test_search_rubrics(self):
        """Test searching rubrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = RubricRepository(Path(tmpdir))

            # Save rubrics with different names and tags
            for name, tags in [
                ("Math Rubric", ["math", "presentation"]),
                ("English Rubric", ["english", "writing"]),
                ("Science Lab Report", ["science", "report"]),
            ]:
                criterion = RubricCriterion(name="Test")
                category = RubricCategory(name="Category", criteria=[criterion])
                rubric = Rubric(name=name, categories=[category], tags=tags)
                repo.save(rubric)

            # Search by name
            results = repo.search("math")
            assert len(results) == 1
            assert results[0].name == "Math Rubric"

            # Search by tag
            results = repo.search("science")
            assert len(results) == 1

            # Search returns multiple matches
            results = repo.search("rubric")
            assert len(results) == 2


class TestRubricScorer:
    """Tests for RubricScorer helper."""

    def test_score_category(self):
        """Test scoring a single category."""
        criterion1 = RubricCriterion(name="Criterion 1", weight=1.0)
        criterion2 = RubricCriterion(name="Criterion 2", weight=2.0)
        category = RubricCategory(name="Content", criteria=[criterion1, criterion2])
        rubric = Rubric(name="Test", categories=[category])

        scorer = RubricScorer(rubric)

        # Score: criterion1=3, criterion2=5
        # Weighted: (3*1 + 5*2) / (1+2) = 13/3 = 4.33
        scores = scorer.score_category(
            category.id, {criterion1.id: 3, criterion2.id: 5}
        )

        assert scores is not None
        assert abs(scores.category_total - 4.33) < 0.01
        assert len(scores.criterion_scores) == 2

    def test_score_all_categories(self):
        """Test scoring all categories."""
        criterion1 = RubricCriterion(name="Criterion 1")
        category1 = RubricCategory(name="Content", weight=1.0, criteria=[criterion1])

        criterion2 = RubricCriterion(name="Criterion 2")
        category2 = RubricCategory(name="Delivery", weight=1.0, criteria=[criterion2])

        rubric = Rubric(name="Test", categories=[category1, category2])
        scorer = RubricScorer(rubric)

        # Both categories scored 4 out of 5
        all_scores = {
            category1.id: {criterion1.id: 4},
            category2.id: {criterion2.id: 4},
        }

        assessment = scorer.score_all_categories(all_scores)

        assert isinstance(assessment, RubricAssessment)
        assert len(assessment.category_scores) == 2
        assert assessment.overall_score == 4.0  # (4+4)/2
        assert assessment.rubric_name == "Test"

    def test_assessment_percentage_calculation(self):
        """Test that percentage is calculated correctly."""
        criterion = RubricCriterion(name="Test")
        category = RubricCategory(name="Category", criteria=[criterion])
        rubric = Rubric(name="Test", categories=[category])

        scorer = RubricScorer(rubric)
        scores = {category.id: {criterion.id: 3}}
        # 1-5 scale, score 3 is middle (50%)
        assessment = scorer.score_all_categories(scores)

        # (3-1)/(5-1) * 100 = 50%
        assert abs(assessment.overall_percentage - 50.0) < 1.0

    def test_assessment_with_weighted_categories(self):
        """Test assessment with weighted categories."""
        criterion1 = RubricCriterion(name="Criterion 1")
        category1 = RubricCategory(
            name="Content", weight=2.0, criteria=[criterion1]
        )  # Worth 2x

        criterion2 = RubricCriterion(name="Criterion 2")
        category2 = RubricCategory(
            name="Delivery", weight=1.0, criteria=[criterion2]
        )  # Worth 1x

        rubric = Rubric(name="Test", categories=[category1, category2])
        scorer = RubricScorer(rubric)

        # Content scores 5, Delivery scores 1
        # Weighted: (5*2 + 1*1) / (2+1) = 11/3 = 3.67
        all_scores = {
            category1.id: {criterion1.id: 5},
            category2.id: {criterion2.id: 1},
        }

        assessment = scorer.score_all_categories(all_scores)
        assert abs(assessment.overall_score - 3.67) < 0.01
