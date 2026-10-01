# Installation

Python 3.10 or newer is required. Make is optional. Verification and offline
tests run directly from the checkout using only the Python standard library.

```sh
make reproduce-main
make test
```

For an installed `tensorbound` command:

```sh
python3 -m venv .work-tmp/venv
. .work-tmp/venv/bin/activate
python3 -m pip install -e .
tensorbound --help
```

Package installation may download build tools. It is optional for reproduction
and for `python3 -m tensorbound` commands.

## Container

```sh
docker build -t tensorbound-artifact .
docker run --rm tensorbound-artifact
```

The default container command verifies the packaged data and prints the
recomputed summary. Building the image requires access to the base image and
Python package index; the verification command itself is offline.

Tests use `.work-tmp/` for temporary data and leave the published data unchanged.
