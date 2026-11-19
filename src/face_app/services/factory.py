"""Monta os serviços a partir da configuração, de um detector e de um extrator."""
from dataclasses import dataclass

from src.face_app.config import Settings
from src.face_app.services.enrollment import EnrollmentService
from src.face_app.services.people import PersonService
from src.face_app.services.recognition import RecognitionService
from src.face_app.storage.repository import PersonRepository


@dataclass(frozen=True)
class Services:
    cadastro: EnrollmentService
    reconhecimento: RecognitionService
    pessoas: PersonService


def build_services(settings: Settings, detector, embedder) -> Services:
    repo = PersonRepository(settings.people_dir)
    args = (detector, embedder, repo, settings.threshold,
            settings.min_confidence, settings.min_face_px)
    return Services(
        cadastro=EnrollmentService(*args),
        reconhecimento=RecognitionService(*args),
        pessoas=PersonService(repo),
    )