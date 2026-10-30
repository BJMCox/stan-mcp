import asyncio


def test_safe_catches_generic_exceptions_as_error_dict():
    from stan_mcp import server

    def boom():
        raise ValueError("boom")

    assert server._safe(boom) == {"error": "boom", "error_type": "ValueError"}


def test_density_tools_registered():
    from stan_mcp import server

    names = asyncio.run(server.mcp.list_tools())
    tool_names = {t.name for t in names}
    for expected in [
        "compile_model",
        "param_info",
        "log_density",
        "log_density_gradient",
        "log_density_hessian",
        "param_constrain",
        "param_unconstrain",
        "check_gradient",
    ]:
        assert expected in tool_names


def test_run_tools_registered():
    import asyncio

    from stan_mcp import server

    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    for expected in [
        "start_run",
        "get_run_status",
        "await_run",
        "list_runs",
        "get_run_summary",
        "get_run_diagnostics",
        "extract_draws",
        "evaluate_batch",
        "compare_batches",
        "get_batch_result",
        "resume_batch",
    ]:
        assert expected in names
