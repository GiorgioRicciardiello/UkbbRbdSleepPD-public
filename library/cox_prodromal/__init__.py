"""
Cox Prodromal Analysis Package.

Modular survival analysis for RBD x prodromal marker associations
with incident neurological outcomes in the UK Biobank.
"""
from library.cox_prodromal.runner import run_prodromal_pipeline

__all__ = ["run_prodromal_pipeline"]
