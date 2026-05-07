"""Custom evaluation rubric system for presentation assessment."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class RubricCriterion(BaseModel):
    """A single evaluation criterion within a rubric category."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique criterion ID"
    )
    name: str = Field(..., min_length=1, max_length=200, description="Criterion name")
    description: str = Field(
        default="", max_length=1000, description="Detailed criterion description"
    )
    weight: float = Field(
        default=1.0, ge=0.0, le=10.0, description="Relative weight/importance"
    )
    scoring_guide: str | None = Field(
        None, max_length=1000, description="Guide for scoring this criterion"
    )
    level_descriptions: dict[int, str] = Field(
        default_factory=dict,
        description="Descriptions for each score level (e.g., {1: 'Poor', 5: 'Excellent'})",
    )


class RubricCategory(BaseModel):
    """A category of criteria within a rubric (e.g., 'Content', 'Delivery')."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique category ID"
    )
    name: str = Field(..., min_length=1, max_length=100, description="Category name")
    description: str | None = Field(
        None, max_length=500, description="Category description"
    )
    weight: float = Field(
        default=1.0, ge=0.0, le=10.0, description="Category weight in overall score"
    )
    criteria: list[RubricCriterion] = Field(
        default_factory=list, description="Evaluation criteria in this category"
    )

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: list[RubricCriterion]) -> list[RubricCriterion]:
        """Ensure at least one criterion per category."""
        if not v:
            raise ValueError("Category must have at least one criterion")
        return v


class ScoringScale(BaseModel):
    """Definition of a scoring scale (e.g., 1-5 stars, percentage)."""

    id: str = Field(default="default", description="Scale identifier")
    name: str = Field(..., description="Scale name (e.g., '1-5 Stars', 'Percentage')")
    min_score: int = Field(..., ge=0, description="Minimum score value")
    max_score: int = Field(..., ge=1, description="Maximum score value")
    default_score: int | None = Field(
        None, description="Default score if not specified"
    )
    labels: dict[int, str] = Field(
        default_factory=dict,
        description="Score labels (e.g., {1: 'Poor', 5: 'Excellent'})",
    )

    @field_validator("max_score")
    @classmethod
    def validate_max_score(cls, v: int, info: Any) -> int:
        """Ensure max_score > min_score."""
        if "min_score" in info.data and v <= info.data["min_score"]:
            raise ValueError("max_score must be greater than min_score")
        return v


class Rubric(BaseModel):
    """Complete evaluation rubric with categories and criteria."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique rubric ID"
    )
    name: str = Field(..., min_length=1, max_length=200, description="Rubric name")
    description: str | None = Field(
        None, max_length=1000, description="Rubric description"
    )
    categories: list[RubricCategory] = Field(
        ..., description="Evaluation categories", min_length=1
    )
    scoring_scale: ScoringScale = Field(
        default_factory=lambda: ScoringScale(
            name="1-5 Scale",
            min_score=1,
            max_score=5,
            default_score=None,
            labels={1: "Poor", 5: "Excellent"},
        ),
        description="Scoring scale for this rubric",
    )
    is_template: bool = Field(
        default=False, description="Whether this is a reusable template"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for organization (e.g., 'Physics 101', 'Final Project')",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation time"
    )
    modified_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last modification time",
    )
    created_by: str | None = Field(None, description="User who created this rubric")
    version: int = Field(default=1, ge=1, description="Rubric version")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump(mode="json")
        data["created_at"] = self.created_at.isoformat()
        data["modified_at"] = self.modified_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rubric":
        """Create from dictionary."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("modified_at"), str):
            data["modified_at"] = datetime.fromisoformat(data["modified_at"])
        return cls(**data)


class RubricScore(BaseModel):
    """Score for a single criterion."""

    criterion_id: str = Field(..., description="ID of criterion being scored")
    score: int = Field(..., description="Score value")
    notes: str | None = Field(None, max_length=500, description="Evaluator notes")
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence for this score (e.g., timestamps, quotes)",
    )


class RubricCategoryScore(BaseModel):
    """Aggregated score for a category."""

    category_id: str = Field(..., description="ID of category")
    category_name: str = Field(..., description="Category name")
    criterion_scores: list[RubricScore] = Field(
        ..., description="Scores for each criterion"
    )
    category_total: float = Field(..., description="Weighted total for category")
    category_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage for category"
    )


class RubricAssessment(BaseModel):
    """Assessment result using a rubric."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique assessment ID"
    )
    rubric_id: str = Field(..., description="ID of rubric used")
    rubric_name: str = Field(..., description="Name of rubric used")
    analysis_id: str | None = Field(
        None, description="ID of video analysis being assessed"
    )
    assessed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Assessment time",
    )
    assessed_by: str | None = Field(None, description="User who performed assessment")
    category_scores: list[RubricCategoryScore] = Field(
        ..., description="Scores by category"
    )
    overall_score: float = Field(..., description="Weighted overall score")
    overall_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Overall percentage score"
    )
    feedback: str | None = Field(None, max_length=2000, description="Overall feedback")
    recommendations: list[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )


class RubricBuilder:
    """Helper class for building rubrics programmatically."""

    def __init__(self, name: str, description: str = ""):
        """Initialize builder."""
        self.name = name
        self.description = description
        self.categories: list[RubricCategory] = []
        self.scoring_scale = ScoringScale(
            name="1-5 Scale",
            min_score=1,
            max_score=5,
            default_score=None,
            labels={1: "Poor", 5: "Excellent"},
        )
        self.created_by: str | None = None

    def add_category(
        self, name: str, description: str = "", weight: float = 1.0
    ) -> "CategoryBuilder":
        """Add a category to the rubric."""
        category = RubricCategory(
            name=name, description=description or None, weight=weight
        )
        self.categories.append(category)
        return CategoryBuilder(category)

    def set_scoring_scale(
        self, min_score: int, max_score: int, labels: dict[int, str] | None = None
    ) -> None:
        """Set custom scoring scale."""
        self.scoring_scale = ScoringScale(
            name=f"{min_score}-{max_score} Scale",
            min_score=min_score,
            max_score=max_score,
            default_score=None,
            labels=labels or {},
        )

    def build(self, is_template: bool = False, tags: list[str] | None = None) -> Rubric:
        """Build the final rubric."""
        return Rubric(
            name=self.name,
            description=self.description or None,
            categories=self.categories,
            scoring_scale=self.scoring_scale,
            is_template=is_template,
            tags=tags or [],
            created_by=self.created_by,
        )


class CategoryBuilder:
    """Helper for building a rubric category."""

    def __init__(self, category: RubricCategory):
        """Initialize with a category."""
        self.category = category

    def add_criterion(
        self,
        name: str,
        description: str = "",
        weight: float = 1.0,
        scoring_guide: str = "",
    ) -> "CategoryBuilder":
        """Add a criterion to the category."""
        criterion = RubricCriterion(
            name=name,
            description=description or "",
            weight=weight,
            scoring_guide=scoring_guide or None,
        )
        self.category.criteria.append(criterion)
        return self


class RubricRepository:
    """Simple file-based repository for storing and retrieving rubrics."""

    def __init__(self, storage_dir: Path):
        """Initialize repository."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"RubricRepository initialized at {self.storage_dir}")

    def save(self, rubric: Rubric) -> None:
        """Save a rubric to disk."""
        rubric_file = self.storage_dir / f"{rubric.id}.json"
        rubric.modified_at = datetime.now(UTC)

        with open(rubric_file, "w") as f:
            json.dump(rubric.to_dict(), f, indent=2)

        logger.info(f"Saved rubric: {rubric.name} ({rubric.id})")

    def load(self, rubric_id: str) -> Rubric | None:
        """Load a rubric by ID."""
        rubric_file = self.storage_dir / f"{rubric_id}.json"

        if not rubric_file.exists():
            return None

        with open(rubric_file) as f:
            data = json.load(f)

        return Rubric.from_dict(data)

    def list_rubrics(self, template_only: bool = False) -> list[Rubric]:
        """List all stored rubrics."""
        rubrics: list[Rubric] = []

        for rubric_file in self.storage_dir.glob("*.json"):
            with open(rubric_file) as f:
                data = json.load(f)

            rubric = Rubric.from_dict(data)

            if not template_only or rubric.is_template:
                rubrics.append(rubric)

        return sorted(rubrics, key=lambda r: r.created_at, reverse=True)

    def delete(self, rubric_id: str) -> bool:
        """Delete a rubric."""
        rubric_file = self.storage_dir / f"{rubric_id}.json"

        if rubric_file.exists():
            rubric_file.unlink()
            logger.info(f"Deleted rubric: {rubric_id}")
            return True

        return False

    def search(self, query: str) -> list[Rubric]:
        """Search rubrics by name or tags."""
        results: list[Rubric] = []
        query_lower = query.lower()

        for rubric in self.list_rubrics():
            if (
                query_lower in rubric.name.lower()
                or query_lower in (rubric.description or "").lower()
                or any(query_lower in tag.lower() for tag in rubric.tags)
            ):
                results.append(rubric)

        return results


class RubricScorer:
    """Helper for scoring presentations against a rubric."""

    def __init__(self, rubric: Rubric):
        """Initialize with a rubric."""
        self.rubric = rubric

    def score_category(
        self, category_id: str, criterion_scores: dict[str, int]
    ) -> RubricCategoryScore | None:
        """Score a single category."""
        # Find category
        category = None
        for cat in self.rubric.categories:
            if cat.id == category_id:
                category = cat
                break

        if not category:
            return None

        # Build scores
        scores: list[RubricScore] = []
        total_weight = 0
        weighted_sum = 0

        for criterion in category.criteria:
            if criterion.id in criterion_scores:
                score_value = criterion_scores[criterion.id]
                scores.append(
                    RubricScore(
                        criterion_id=criterion.id,
                        score=score_value,
                        notes=None,
                    )
                )

                weighted_sum += score_value * criterion.weight
                total_weight += criterion.weight

        # Calculate category scores
        category_total = weighted_sum / total_weight if total_weight > 0 else 0
        scale_range = (
            self.rubric.scoring_scale.max_score - self.rubric.scoring_scale.min_score
        )
        category_percentage = (
            ((category_total - self.rubric.scoring_scale.min_score) / scale_range * 100)
            if scale_range > 0
            else 0
        )

        return RubricCategoryScore(
            category_id=category.id,
            category_name=category.name,
            criterion_scores=scores,
            category_total=category_total,
            category_percentage=max(0, min(100, category_percentage)),
        )

    def score_all_categories(
        self, all_scores: dict[str, dict[str, int]]
    ) -> RubricAssessment:
        """Score all categories and generate assessment."""
        category_scores: list[RubricCategoryScore] = []
        total_weight = 0
        weighted_sum = 0

        for category in self.rubric.categories:
            cat_score = self.score_category(
                category.id, all_scores.get(category.id, {})
            )
            if cat_score:
                category_scores.append(cat_score)
                weighted_sum += cat_score.category_total * category.weight
                total_weight += category.weight

        # Calculate overall score
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        scale_range = (
            self.rubric.scoring_scale.max_score - self.rubric.scoring_scale.min_score
        )
        overall_percentage = (
            ((overall_score - self.rubric.scoring_scale.min_score) / scale_range * 100)
            if scale_range > 0
            else 0
        )

        return RubricAssessment(
            rubric_id=self.rubric.id,
            rubric_name=self.rubric.name,
            analysis_id=None,
            assessed_by=None,
            category_scores=category_scores,
            overall_score=overall_score,
            overall_percentage=max(0, min(100, overall_percentage)),
            feedback=None,
        )
