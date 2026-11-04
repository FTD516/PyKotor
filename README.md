# PyKotor

Application-scoped resource, patching and rendering libraries used by
HoloPatcher and Holocron Toolset for Knights of the Old Republic and
The Sith Lords. This tree is not a drop-in replacement for every API in the
full upstream PyKotor source.

Python **3.10 or newer** is required. CI targets 3.10–3.13 on Windows, Linux and
macOS for the core suite.

Support is offered in the OpenKotOR Discord.

[![OpenKotOR Discord](https://discordapp.com/api/guilds/739590575359262792/widget.png?style=banner2)](https://discord.gg/OpenKotOR)

## Source development

Create/activate a virtual environment, then run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -ra
python -m ruff check Libraries tests
```

For offline development, provide the dependency wheels locally:

```bash
python -m pip install --no-index --find-links /path/to/wheels -r requirements-dev.txt
```

Pytest registers the three source directories through the root `pyproject.toml`.
Ruff is the only configured formatter/linter. The CI gate checks syntax errors,
undefined names and other fatal Python errors without rewriting legacy style.
`ruff format` may be used deliberately on files being edited.


## Building distributable libraries

With `build`, setuptools and wheel already installed, run from this root:

```bash
python -m build --no-isolation --outdir dist Libraries/Utility
python -m build --no-isolation --outdir dist Libraries/PyKotor
python -m build --no-isolation --outdir dist Libraries/PyKotorGL
```

The commands use each library's explicit PEP 517/621 metadata, not a custom
`setup.py` parser. Repository test packages are not included in the wheels, and
the core and GL wheels do not overlap in owned files.

For offline installation into a separate environment, include these three matching wheels and the necessary third-party wheels in the local wheel directory:

```bash
python -m pip install --no-index --find-links /path/to/wheels PyKotorUtility==1.7 PyKotor==1.7 PyKotorGL==1.7
python -m pip check
```

## Standalone applications

Set `PYKOTOR_ROOT` to this repository and run either frontend's `run.py --backend-info` to verify selection. Application-specific
`HOLOPATCHER_PYKOTOR_ROOT` and `HOLOCRON_PYKOTOR_ROOT` overrides take precedence.

