# NLP Pandas

It's an extension for pandas providing some NLP functionalities for strings.

[![build](https://github.com/jaume-ferrarons/pandas-nlp/actions/workflows/push-event.yml/badge.svg?branch=master)](https://github.com/jaume-ferrarons/pandas-nlp/actions/workflows/push-event.yml)
[![version](https://img.shields.io/pypi/v/pandas_nlp?logo=pypi&logoColor=white)](https://pypi.org/project/pandas-nlp/)

## Installation

Install with:
```bash
pip install -U pandas-nlp
```

### Requirements 
- python >= 3.8

## Key features

### Language detection
```python
import pandas as pd
import pandas_nlp

df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "text": [
        "I like cats",
        "Me gustan los gatos",
        "M'agraden els gats",
        "J'aime les chats",
        "Ich mag Katzen",
    ],
})
df.text.nlp.language()
```
**Output**
```
0    en
1    es
2    ca
3    fr
4    de
Name: text_language, dtype: object
```

### String embedding
```python
import pandas as pd
import pandas_nlp

df = pd.DataFrame(
    {"id": [1, 2, 3], "text": ["cat", "dog", "violin"]}
)
df.text.nlp.embedding()
```
**Output**
```
0    [2.0860276, 0.78038394, 0.20159146, -1.2828196...
1    [0.96052396, 1.0350337, 0.11549556, -1.2252672...
2    [1.2934866, 0.10021937, 0.71453714, -1.3288003...
Name: text_embedding, dtype: object
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
0    [Hello, how are you?]
1     [Code., Sleep., Eat]
Name: text_sentences, dtype: object
```