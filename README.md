# The G(TE)<sup>2</sup> Service


The **G(TE)<sup>2</sup>** (Glossary Term Exploration and Extraction) service is a tool that is capable of extracting glossary terms from German Requirements sentences and delivering statistical indicators for ranking as well as clustering labels. The service uses a local requirements database to generate parts of the statistical indicators. The tool additionally links each glossary term candidate to its definition from a local definitions database. 

Since we are not allowed to publish the definition and reference datasets, we use a open-source requirements dataset named [OPENCOSS](https://sites.google.com/site/svvregice/evaluation) to evaluate the results. 


## Research Contributions

+ Automatic detection of glossary terms from German requirements documents using [Python](https://python.org)-built technology ([spaCy](https://spacy.io/), [CharSplit](https://github.com/dtuggener/CharSplit))
+ Clustering of glossary terms using a keyword-based approach
+ Evaluation of glossary clustering results with the [Omega index](https://github.com/isaranto/omega_index) (tests and results on `../clustering-optimization`)

## Features

+ Automatic glossary terms candidates elicitation from a requirements document using spaCy Part-of-Speech tagging
+ Clustering of detected glossary candidates into disjoint partitions using state of the art clustering algorithms and vectorization techniques (soft clustering tested on `../clustering-optimization`)
+ Generation of statistical indicators for glossary candidates relevancy ranking



## Expected Input

The server expect the sentences to folow the JSON format. An example is outlined below:

```json
[
  {
    "id": "1",
    "text": "Das System muss Schlüsselbegriffe durch statistische Auswertungen zur Eintragung ins Glossar vorschlagen."
  }
]
```

## Code Base

The server is located in `/glossar-term-extraction-service`.


