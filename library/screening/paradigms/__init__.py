"""Training paradigm registry."""
from library.screening.paradigms.p1_combined import CombinedTrainingParadigm
from library.screening.paradigms.p2_incident_only import IncidentOnlyParadigm
from library.screening.paradigms.p3_weighted import WeightedTrainingParadigm
from library.screening.paradigms.p4_subsampling import SubsamplingParadigm
from library.screening.paradigms.p6_prevalent_train import PrevalentTrainParadigm

__all__ = [
    "CombinedTrainingParadigm",
    "IncidentOnlyParadigm",
    "WeightedTrainingParadigm",
    "SubsamplingParadigm",
    "PrevalentTrainParadigm",
]
