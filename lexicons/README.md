# Extra lexicons for the Docker image

Put generated lexicon files here (for example `nia.csv` from `scripts/import_nia_terminologies.py`)
and build with:

    docker build --build-arg AYURNER_LEXICONS=/app/lexicons/nia.csv -t ayurner .

This folder is copied into the image. Check a source's licence before committing its data.
