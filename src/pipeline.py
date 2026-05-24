"""Main pipeline orchestrator."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STAGE_NAMES = {
    1: "Ingest",
    2: "Classify",
    3: "Chunk",
    4: "Summarize",
    5: "Extract Doctrine",
    6: "Assemble",
    7: "Report",
}


def run(cfg: dict, from_stage: int = 1, only_stage: int | None = None, force: bool = False) -> None:
    stages = [only_stage] if only_stage else list(range(from_stage, 8))

    for stage_num in stages:
        name = STAGE_NAMES.get(stage_num, f"Stage {stage_num}")
        logger.info(f"\n{'='*50}")
        logger.info(f"Stage {stage_num}: {name}")
        logger.info(f"{'='*50}")

        if stage_num == 1:
            from src.stages.s1_ingest import run as s1
            s1(cfg, force=force)
        elif stage_num == 2:
            from src.stages.s2_classify import run as s2
            s2(cfg, force=force)
        elif stage_num == 3:
            from src.stages.s3_chunk import run as s3
            s3(cfg, force=force)
        elif stage_num == 4:
            from src.stages.s4_summarize import run as s4
            s4(cfg, force=force)
        elif stage_num == 5:
            from src.stages.s5_extract import run as s5
            s5(cfg, force=force)
        elif stage_num == 6:
            from src.stages.s6_assemble import run as s6
            s6(cfg, force=force)
        elif stage_num == 7:
            from src.stages.s7_report import run as s7
            s7(cfg)
        else:
            logger.error(f"Unknown stage: {stage_num}")
