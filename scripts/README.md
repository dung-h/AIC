# Operational scripts

`competition.sh` is the supported operational boundary:

```bash
./scripts/competition.sh preflight
./scripts/competition.sh bootstrap plan
./scripts/competition.sh run --queries QUERY.json --output OUT.json
```

It resolves a native-Linux interpreter, applies one safe shared `.env`
contract (explicit exports override file values), applies offline Hugging Face
defaults, and delegates to the fail-closed CLI modules under `src/cli/`.

`runtime_data_bootstrap.py` is explicit opt-in: `plan` never uses the network;
only `fetch --yes` downloads or extracts artifacts. `build_source_bundle.sh`
creates a source-only tarball from the same Git ignore contract. Benchmark and
profiling scripts are diagnostic, not submission entrypoints.
