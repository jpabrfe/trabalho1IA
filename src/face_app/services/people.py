"""Listagem e exclusão de pessoas (spec 006), para a interface."""
from src.face_app.storage.repository import Person, PersonRepository


class PersonService:
    def __init__(self, repo: PersonRepository):
        self._repo = repo

    def listar(self) -> list[Person]:
        return self._repo.list_people()

    def contar(self) -> int:
        return self._repo.count()

    def excluir(self, id_) -> bool:
        return self._repo.excluir(id_)