import asyncio


def test_density_tools_registered():
    import server

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

    import server

    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    for expected in [
        "start_run",
        "get_run_status",
        "await_run",
        "list_runs",
        "get_run_summary",
        "get_run_diagnostics",
        "extract_draws",
    ]:
        assert expected in names
