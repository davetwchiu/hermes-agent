# Gooaye production scripts

These scripts and their Python import modules ship with the immutable Hermes release. Their Python dependencies use the repository's `uv.lock` (SHA-256 recorded in `runtime-pins.json`). Node, FFmpeg, and ffprobe are pinned by version and binary SHA-256 there and checked before use.

The vendored receipt helper is the byte-exact `.github/scripts/vault_curation_reconcile.cjs` from `davetwchiu/david-knowledge-vault` main commit `32b5b5e81f9ea2018123ceafc11fb6712d067e43` (Git blob `9171c69ec84e36e68dc4502f259a4e23594ed110`). Its runtime path is fixed beside these scripts and its SHA-256 is checked before each receipt validation.

The scripts keep runtime state under the active `HERMES_HOME/state/`; secrets remain in the profile secret store. No episode ledger, transcript, cache, or credential belongs in this directory.
