try:
    import reaktoro  # optional heavy dep
except Exception:
    reaktoro = None

from sep_agents.dsl.schemas import Stream

async def run_speciation(params: dict):
    """
    params = {
      "stream": { ... schema-compatible Stream dict ... }
    }
    """
    if reaktoro is None:
        return {"status": "error", "error": "Reaktoro not installed. Install with `pip install .[thermo]` or `conda install -c conda-forge reaktoro`."}

    s = Stream(**params["stream"])

    # TODO: Build a Reaktoro system, equilibrate, and map back to Stream.
    # Placeholder: echo input
    return {"status": "ok", "stream_out": s.dict()}
