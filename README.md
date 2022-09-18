# NLP Pandas

It's an extension for pandas providing some NLP functionalities for strings.

[![build](https://github.com/jaume-ferrarons/pandas-nlp/actions/workflows/push-event.yml/badge.svg?branch=master)](https://github.com/jaume-ferrarons/pandas-nlp/actions/workflows/push-event.yml)
[![version](https://img.shields.io/pypi/v/pandas_nlp?logo=pypi&logoColor=white)](https://pypi.org/project/pandas-nlp/)

## Installation

Install with:
```bash
pip install -U pandas-nlp
```

## Key features
### String embedding
```python
import pandas as pd
import pandas_nlp

df = pd.DataFrame(
    {"id": [1, 2, 3], "text": ["cat", "dog", "violin"]}
)
df.text.nlp.embedding()
```

### String embedding
```python
import pandas as pd
import pandas_nlp

df = pd.DataFrame(
    {"id": [0, 1], "text": ["Hello, how are you?", "Code. Sleep. Eat"]}
)
df.text.nlp.sentences()
```
**Output**
```python
[['Hello, how are you?'], ['Code.', 'Sleep.', 'Eat']]
```