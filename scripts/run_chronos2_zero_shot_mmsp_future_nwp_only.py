#!/usr/bin/env python3
"""Run Chronos-2 with FusionSF-equivalent future-only NWP information.

This is intentionally a separate entry point.  The original
``run_chronos2_zero_shot_mmsp.py`` behavior and its existing outputs remain
unchanged.

Chronos-2 requires every key in ``future_covariates`` to also occur in
``past_covariates``.  To expose only the future NWP values used by FusionSF,
the corresponding past covariates are supplied as NaN (Chronos' missing-value
marker), rather than as observed past NWP values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import run_chronos2_zero_shot_mmsp as original


DEFAULT_OUTPUT_DIR = original.ROOT / "logs/chronos2_future_nwp_only_mmsp_site0_9"


class FutureOnlyNWPCovariates(original.MMSPNWPCovariates):
    """Construct Chronos inputs with real future NWP and missing past NWP."""

    def chronos_input(
        self, global_window: int, windows_per_site: int, target: np.ndarray
    ) -> dict:
        site = global_window // windows_per_site
        local_window = global_window % windows_per_site
        start = self.test_start + local_window
        middle = start + self.context_length
        stop = middle + self.prediction_length
        nwp = self.groups[self.site_coords[site]]
        if stop > len(nwp):
            raise IndexError(
                f"NWP slice exceeds available data for site {site}: {stop}>{len(nwp)}"
            )

        # Chronos-2 requires future-covariate keys to be a subset of the
        # past-covariate keys. NaN marks those past values as unavailable.
        missing_past = np.full(self.context_length, np.nan, dtype=np.float32)
        return {
            "target": target,
            "past_covariates": {
                name: missing_past.copy() for name in self.feature_names
            },
            "future_covariates": {
                name: nwp[middle:stop, index]
                for index, name in enumerate(self.feature_names)
            },
        }


def requested_output_dir(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args, _ = parser.parse_known_args(argv)
    return args.output_dir


def main() -> int:
    output_dir = requested_output_dir(sys.argv[1:])

    # Reuse the validated data loading, inference, and persistence path while
    # replacing only the covariate construction and default output location.
    original.MMSPNWPCovariates = FutureOnlyNWPCovariates
    original.DEFAULT_NWP_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    if "--use-nwp" not in sys.argv:
        sys.argv.append("--use-nwp")

    result = original.main()

    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    loaded_model_reference = metadata["model"]
    if "models--amazon--chronos-2" in loaded_model_reference:
        metadata["model"] = "amazon/chronos-2"
        metadata["model_loaded_from"] = loaded_model_reference
    metadata.update(
        {
            "nwp_context_policy": "future_only",
            "past_nwp_values_provided": False,
            "past_nwp_placeholder": "nan",
            "future_nwp_values_provided": True,
            "fusionSF_equivalent_nwp_horizon": True,
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"updated_metadata={metadata_path.resolve()}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
