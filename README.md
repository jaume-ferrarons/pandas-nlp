# Pandas NLP

It's an extension for pandas providing some NLP functionalities for strings.

[![build](https://github.com/jaume-ferrarons/pandas-nlp/actions/workflows/push-event.yml/badge.svg?branch=master)](https://github.com/jaume-ferrarons/pandas-nlp/actions/workflows/push-event.yml)
[![version](https://img.shields.io/pypi/v/pandas_nlp?logo=pypi&logoColor=white)](https://pypi.org/project/pandas-nlp/)
[![codecov](https://codecov.io/gh/jaume-ferrarons/pandas-nlp/branch/master/graph/badge.svg?token=UQUSYGANFQ)](https://codecov.io/gh/jaume-ferrarons/pandas-nlp)
[![pyversion-button](https://img.shields.io/pypi/pyversions/pandas_nlp.svg)](https://pypi.org/project/pandas-nlp/)
## Setup
### Requirements 
- python >= 3.8

### Installation
Execute:
```bash
pip install -U pandas-nlp
```
To install the default spacy English model:
```bash
spacy install en_core_web_md
```


## Key features

### Language detection
```python
import pandas as pd
import pandas_nlp

pandas_nlp.register()

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
with confidence:
```python
df.text.nlp.language(confidence=True).apply(pd.Series)
```
**Output**
```
  language  confidence
0       en    0.897090
1       es    0.982045
2       ca    0.999806
3       fr    0.999713
4       de    0.997995
```

### String embedding
```python
import pandas as pd
import pandas_nlp

pandas_nlp.register()

df = pd.DataFrame(
    {"id": [1, 2, 3], "text": ["cat", "dog", "violin"]}
)
df.text.nlp.embedding()
```
**Output**
```
0    [3.7032, 4.1982, -5.0002, -11.322, 0.031702, -...
1    [1.233, 4.2963, -7.9738, -10.121, 1.8207, 1.40...
2    [-1.4708, -0.73871, 0.49911, -2.1762, 0.56754,...
Name: text_embedding, dtype: object
```

### Closest concept
```python
import pandas as pd
import pandas_nlp

pandas_nlp.register()

themed = pd.DataFrame({
    "id": [0, 1, 2, 3],
    "text": [
        "My computer is broken",
        "I went to a piano concert",
        "Chocolate is my favourite",
        "Mozart played the piano"
    ]
})

themed.text.nlp.closest(["music", "informatics", "food"])
```
**Output**
```
0    informatics
1          music
2           food
3          music
Name: text_closest, dtype: object
```

### Sentence extraction
```python
import pandas as pd
import pandas_nlp

pandas_nlp.register()

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