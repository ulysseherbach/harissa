# Building the documentation

Use python `3.10` or `3.11` but not `3.12` because `sphinxcontrib.collections`
depends on something that is deprecated and removed in `3.12`.

## Install dependencies

From the `docs` folder of the project, you can install the dependencies required for the 
documentation generation by typing:

```bash
pip install -r requirements.txt
```

## Build website pages

To build the website run the command:

```bash
sphinx-build . _build
```

## Visualize the website

You can open the `_build/index.html` file in your browser.
