"""Evaluate raw microsoft/phi-4 (zero fine-tuning) on both languages.

Establishes the control baseline by running the full evaluation suite
on the unmodified Phi-4 model for Kannada and Telugu.  Results are
appended to ``results/all_results.json`` and a convenience copy is
written to ``results/baseline_phi4.json``.

Usage:
    python prestudy/base_eval.py
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path for sibling-package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eval_engine import evaluate_checkpoint
from utils.data_utils import load_config
from utils.logging_utils import get_logger, load_results, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Run base-model evaluation for both Dravidian languages.

    Evaluates the raw ``microsoft/phi-4`` checkpoint on Kannada and
    Telugu, persists results to ``all_results.json``, and writes a
    filtered convenience file ``baseline_phi4.json``.
    """
    setup_logging("module_0_prestudy", "module_0_prestudy")
    model_config = load_config("configs/model.yaml")
    data_config = load_config("configs/data.yaml")

    for language in ["kannada", "telugu"]:
        experiment_id = f"prestudy_none_{language}"
        logger.info("Starting base evaluation for %s ...", language)

        result = evaluate_checkpoint(
            checkpoint_path=model_config["model_id"],
            language=language,
            experiment_id=experiment_id,
            model_config=model_config,
            data_config=data_config,
            module=0,
            technique="none",
            load_in_4bit=False,
            train_time_hrs=None,
            notes="Control baseline — raw Phi-4 with zero fine-tuning",
        )

        logger.info(
            "Baseline [%s]: chrF=%.2f, perplexity=%.1f",
            language,
            result["chrf"],
            result["perplexity"],
        )

    # --- Write convenience baseline file -----------------------------------
    local_results_dir = Path("results/module_0_prestudy/results")
    all_results = load_results(str(local_results_dir / "all_results.json"))
    prestudy_entries = [r for r in all_results if r.get("module") == 0]

    baseline_path = local_results_dir / "baseline_phi4.json"
    baseline_path.write_text(
        json.dumps(prestudy_entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Base eval complete. Baseline saved to %s", baseline_path
    )


if __name__ == "__main__":
    main()
