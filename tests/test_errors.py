from src.face_app import errors


def test_erros_herdam_da_base_e_tem_mensagem():
    classes = (
        errors.ConfigError, errors.InvalidNameError, errors.InvalidImageError,
        errors.NoFaceError, errors.MultipleFacesError, errors.DuplicatePersonError,
    )
    for cls in classes:
        e = cls()
        assert isinstance(e, errors.FaceAppError)
        assert e.mensagem and str(e) == e.mensagem
    assert errors.NoFaceError("outra").mensagem == "outra"