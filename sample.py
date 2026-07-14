"""Background run manager for all six CmdStanPy inference methods.

start_run launches a method (sample | optimize | variational | pathfinder |
laplace | generate_quantities) on a ThreadPoolExecutor and returns a run_id
immediately. A run_id -> handle{status, asyncio.Event, result} registry backs
get_run_status (non-blocking) and await_run (blocks server-side on the event and
returns the result the moment the run finishes — the MCP-native "ping").

Implemented during the build (see design tool group C).
"""
