"""Erros da aplicação (specs 002 e 006). A interface mostra `mensagem` ao usuário."""


class FaceAppError(Exception):
    mensagem_padrao = "Erro na aplicação."

    def __init__(self, mensagem=None):
        self.mensagem = mensagem or self.mensagem_padrao
        super().__init__(self.mensagem)


class ConfigError(FaceAppError):
    mensagem_padrao = "Configuração inválida."


class InvalidNameError(FaceAppError):
    mensagem_padrao = "Nome inválido."


class InvalidImageError(FaceAppError):
    mensagem_padrao = "Imagem ilegível ou com face inadequada (pequena demais ou pouco nítida)."


class NoFaceError(FaceAppError):
    mensagem_padrao = "Nenhuma face detectada na imagem."


class MultipleFacesError(FaceAppError):
    mensagem_padrao = "Mais de uma face na imagem; use uma foto com uma só pessoa."


class DuplicatePersonError(FaceAppError):
    mensagem_padrao = "Esta pessoa já está cadastrada."