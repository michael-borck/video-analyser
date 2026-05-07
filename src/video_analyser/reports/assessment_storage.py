"""Persistent storage for assessment reports."""

import json
import logging
from pathlib import Path
from typing import Any

from .assessment_report import AssessmentReport

logger = logging.getLogger(__name__)


class AssessmentStorage:
    """Handle storage and retrieval of assessment reports."""

    def __init__(self, storage_dir: str | Path = "assessments"):
        """
        Initialize assessment storage.

        Args:
            storage_dir: Directory for storing assessment JSON files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"AssessmentStorage initialized at {self.storage_dir}")

    def save(self, assessment: AssessmentReport) -> Path:
        """
        Save assessment to disk.

        Args:
            assessment: AssessmentReport to save

        Returns:
            Path to saved file
        """
        assessment_file = self.storage_dir / f"{assessment.id}.json"

        try:
            with open(assessment_file, "w") as f:
                json.dump(assessment.to_dict(), f, indent=2, default=str)
            logger.info(f"Assessment {assessment.id} saved to {assessment_file}")
            return assessment_file
        except Exception as e:
            logger.error(f"Failed to save assessment {assessment.id}: {e}")
            raise

    def load(self, assessment_id: str) -> AssessmentReport | None:
        """
        Load assessment from disk.

        Args:
            assessment_id: Assessment ID to load

        Returns:
            AssessmentReport or None if not found
        """
        assessment_file = self.storage_dir / f"{assessment_id}.json"

        if not assessment_file.exists():
            logger.warning(f"Assessment file not found: {assessment_file}")
            return None

        try:
            with open(assessment_file) as f:
                data = json.load(f)
            return AssessmentReport.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load assessment {assessment_id}: {e}")
            return None

    def list_assessments(
        self,
        analysis_id: str | None = None,
        assessed_by: str | None = None,
        include_drafts: bool = True,
    ) -> list[AssessmentReport]:
        """
        List assessments with optional filtering.

        Args:
            analysis_id: Filter by analysis ID
            assessed_by: Filter by assessor name
            include_drafts: Include draft assessments

        Returns:
            List of matching assessments
        """
        assessments: list[AssessmentReport] = []

        for assessment_file in self.storage_dir.glob("*.json"):
            try:
                assessment = self.load(assessment_file.stem)
                if assessment is None:
                    continue

                # Apply filters
                if analysis_id and assessment.analysis_id != analysis_id:
                    continue
                if assessed_by and assessment.assessed_by != assessed_by:
                    continue
                if not include_drafts and assessment.is_draft:
                    continue

                assessments.append(assessment)
            except Exception as e:
                logger.warning(f"Failed to load assessment from {assessment_file}: {e}")

        return sorted(assessments, key=lambda a: a.assessed_at, reverse=True)

    def get_latest_assessment(self, analysis_id: str) -> AssessmentReport | None:
        """
        Get the latest assessment for an analysis.

        Args:
            analysis_id: Analysis ID to find assessments for

        Returns:
            Latest assessment or None
        """
        assessments = self.list_assessments(
            analysis_id=analysis_id, include_drafts=False
        )
        return assessments[0] if assessments else None

    def get_draft_assessment(self, analysis_id: str) -> AssessmentReport | None:
        """
        Get the draft assessment for an analysis (if any).

        Args:
            analysis_id: Analysis ID to find draft for

        Returns:
            Draft assessment or None
        """
        assessments = self.list_assessments(
            analysis_id=analysis_id, include_drafts=True
        )
        for assessment in assessments:
            if assessment.is_draft:
                return assessment
        return None

    def delete(self, assessment_id: str) -> bool:
        """
        Delete an assessment.

        Args:
            assessment_id: Assessment ID to delete

        Returns:
            True if deleted, False if not found
        """
        assessment_file = self.storage_dir / f"{assessment_id}.json"

        if not assessment_file.exists():
            logger.warning(f"Assessment not found: {assessment_id}")
            return False

        try:
            assessment_file.unlink()
            logger.info(f"Assessment {assessment_id} deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete assessment {assessment_id}: {e}")
            raise

    def search_by_video(self, video_path: str) -> list[AssessmentReport]:
        """
        Find all assessments for a specific video.

        Args:
            video_path: Video file path to search for

        Returns:
            List of assessments for this video
        """
        video_path = str(video_path)
        assessments: list[AssessmentReport] = []

        for assessment_file in self.storage_dir.glob("*.json"):
            try:
                assessment = self.load(assessment_file.stem)
                if assessment and assessment.video_file_path == video_path:
                    assessments.append(assessment)
            except Exception as e:
                logger.warning(f"Failed to load assessment from {assessment_file}: {e}")

        return sorted(assessments, key=lambda a: a.assessed_at, reverse=True)

    def get_assessments_by_assessor(self, assessor_name: str) -> list[AssessmentReport]:
        """
        Get all assessments by a specific assessor.

        Args:
            assessor_name: Assessor name/ID

        Returns:
            List of assessments by this assessor
        """
        return self.list_assessments(assessed_by=assessor_name, include_drafts=False)

    def export_statistics(self, analysis_id: str | None = None) -> dict[str, Any]:
        """
        Generate statistics about assessments.

        Args:
            analysis_id: Optional filter by analysis ID

        Returns:
            Statistics dictionary
        """
        assessments = self.list_assessments(
            analysis_id=analysis_id, include_drafts=False
        )

        if not assessments:
            return {
                "total_assessments": 0,
                "average_score": 0.0,
                "scores": [],
            }

        scores: list[float] = []
        assessors: set[str] = set()

        for assessment in assessments:
            if assessment.rubric_application:
                scores.append(assessment.rubric_application.overall_percentage)
            assessors.add(assessment.assessed_by)

        return {
            "total_assessments": len(assessments),
            "unique_assessors": len(assessors),
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "scores": scores,
            "assessors": sorted(assessors),
        }
