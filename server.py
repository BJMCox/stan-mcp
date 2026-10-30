"""Compatibility launcher; prefer ``python -m stan_mcp``."""

if __name__ == "__main__":
    from stan_mcp.server import main

    main()
