# Operational scripts

`competition.sh` is the supported operational boundary:

```bash
./scripts/competition.sh preflight
./scripts/competition.sh bootstrap plan
./scripts/competition.sh public-bootstrap plan
./scripts/competition.sh run --queries QUERY.json --output OUT.json
```

It resolves a native-Linux interpreter, applies one safe shared `.env`
contract (explicit exports override file values), applies offline Hugging Face
defaults, and delegates to the fail-closed CLI modules under `src/cli/`.

`runtime_data_bootstrap.py` is explicit opt-in: `plan` never uses the network;
only `fetch --yes` downloads or extracts artifacts. `build_source_bundle.sh`
creates a source-only tarball from the same Git ignore contract. Benchmark and
profiling scripts are diagnostic, not submission entrypoints.

`public_runtime_bootstrap.py` is the credential-free equivalent for public
Google Drive folders. It uses `gdown`, requires separate public links for
`data/index`, `data/keyframe_archives_v2`, and `models`, and refuses to merge
over an existing runtime. Its default is the preselection L-series archives
only and the production bge-m3/Qwen 7B models; it never downloads the legacy
raw keyframe tree or Qwen 3B. Use `--asset keyframes --archive <pack>` to add
one archive later without downloading indexes or models again. It must be
invoked as `public-bootstrap fetch --yes`.
