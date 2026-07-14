import errors


def test_as_error_dict_includes_type_and_message():
    d = errors.as_error_dict(errors.ValidationError("bad theta length"))
    assert d == {"error": "bad theta length", "error_type": "ValidationError"}


def test_subclasses_are_stan_mcp_errors():
    assert issubclass(errors.ToolchainError, errors.StanMcpError)
    assert issubclass(errors.CompileError, errors.StanMcpError)
    assert issubclass(errors.RunError, errors.StanMcpError)
