"""Python API package exposing IDTAP data classes and client."""

__version__ = "0.1.50"

from .client import SwaraClient
from .auth import login_google

from .classes.articulation import Articulation
from .classes.automation import Automation  # type: ignore
from .classes.assemblage import Assemblage
from .classes.chikari import Chikari
from .classes.group import Group
from .classes.meter import Meter
from .classes.musical_time import MusicalTime
from .classes.note_view_phrase import NoteViewPhrase
from .classes.piece import Piece
from .classes.phrase import Phrase
from .classes.pitch import Pitch
from .classes.raga import Raga
from .classes.section import Section
from .classes.reconstruct import reconstruct_piece
from .classes.simple_trajectory import (
    OrientationDot,
    SimpleTrajectory,
    decompose_trajectory,
)
from .classes.trajectory import Trajectory

from .enums import Instrument
from .spectrogram import SpectrogramData, SUPPORTED_COLORMAPS
from .audio_models import (
    AudioMetadata,
    AudioUploadResult,
    AudioEventConfig,
    Musician,
    Location,
    RecordingDate,
    Raga as AudioRaga,
    PerformanceSection,
    Permissions,
    ValidationResult,
    LocationHierarchy,
    FileInfo,
    ProcessingStatus
)

# Query system exports
from .query import Query
from .query_types import (
    CategoryType,
    DesignatorType,
    SegmentationType,
    QueryType,
    QueryAnswerType,
    MultipleReturnType,
    MultipleOptionType,
    SecCatType,
    PhraseCatType
)
from .sequence_utils import (
    find_sequence_indexes,
    loose_sequence_indexes,
    split_trajs_by_silences
)

__all__ = [
    "SwaraClient",
    "Articulation",
    "Automation",
    "Assemblage",
    "Chikari",
    "Group",
    "Meter",
    "MusicalTime",
    "NoteViewPhrase",
    "Piece",
    "Phrase",
    "Pitch",
    "Raga",
    "Section",
    "SimpleTrajectory",
    "reconstruct_piece",
    "OrientationDot",
    "decompose_trajectory",
    "Trajectory",
    "Instrument",
    "login_google",
    # Spectrogram
    "SpectrogramData",
    "SUPPORTED_COLORMAPS",
    # Audio upload classes
    "AudioMetadata",
    "AudioUploadResult", 
    "AudioEventConfig",
    "Musician",
    "Location",
    "RecordingDate",
    "AudioRaga",
    "PerformanceSection",
    "Permissions",
    "ValidationResult",
    "LocationHierarchy",
    "FileInfo",
    "ProcessingStatus",
    # Query system
    "Query",
    "CategoryType",
    "DesignatorType", 
    "SegmentationType",
    "QueryType",
    "QueryAnswerType",
    "MultipleReturnType",
    "MultipleOptionType",
    "SecCatType",
    "PhraseCatType",
    "find_sequence_indexes",
    "loose_sequence_indexes",
    "split_trajs_by_silences",
]
