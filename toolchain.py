"""Toolchain provisioning: ensure BridgeStan + CmdStan are present, fail loud."""

from errors import ToolchainError


def _bridgestan_path() -> str:
    # Downloads the pinned BridgeStan source to ~/.bridgestan on first call.
    from bridgestan.compile import get_bridgestan_path

    return str(get_bridgestan_path())


def _cmdstan_path() -> str:
    import cmdstanpy

    return str(cmdstanpy.cmdstan_path())


def _install_cmdstan() -> None:
    import cmdstanpy

    cmdstanpy.install_cmdstan()


def ensure_bridgestan() -> str:
    try:
        return _bridgestan_path()
    except Exception as exc:  # noqa: BLE001 - surfaced as a remediation message
        raise ToolchainError(
            "BridgeStan source tree unavailable and could not be downloaded "
            f"({exc}). Check network access or run "
            "`python -c \"from bridgestan.compile import get_bridgestan_path; "
            'get_bridgestan_path()"` manually.'
        ) from exc


def ensure_cmdstan() -> str:
    try:
        return _cmdstan_path()
    except Exception:
        try:
            _install_cmdstan()
            return _cmdstan_path()
        except Exception as exc:  # noqa: BLE001
            raise ToolchainError(
                "CmdStan not installed and install_cmdstan() failed "
                f"({exc}). Run `python -c \"import cmdstanpy; "
                'cmdstanpy.install_cmdstan()"` manually.'
            ) from exc
