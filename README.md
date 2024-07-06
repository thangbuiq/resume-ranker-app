# resume-ranker-app

A resume ranking app

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

To get the pickled models and functions, run the following command:

```bash
python save_models_and_functions.py
```
Run UI:

```bash
python demo.py
# or
gradio run demo.py
```