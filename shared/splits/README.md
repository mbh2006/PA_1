# Split files (committed)

`pacs_sketch_seed6304.json` is created once by

    python -m shared.build_splits --data-root <pacs root> \
        --out shared/splits/pacs_sketch_seed6304.json

and then committed. Tasks 2 and 3 load it (`--splits`) so that every method and
both tasks see exactly the same source train/validation images. Never regenerate
it with different arguments.
