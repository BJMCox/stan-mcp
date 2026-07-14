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
