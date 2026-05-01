# Raw Financial News Corpus

This directory stores the raw financial-news corpus used by the FinBERT preprocessing pipeline.

The repository tracks the compressed archive through Git LFS:

```text
data/raw/news/All_external.csv.zip
```

Extract it in place before running the sentiment conversion scripts:

```bash
unzip data/raw/news/All_external.csv.zip -d data/raw/news/
```

After extraction, the raw CSV should be available at:

```text
data/raw/news/All_external.csv
```

The local release file used in this project has approximately 13,057,514 rows and the following columns:

```text
Date, Article_title, Stock_symbol, Url, Publisher, Author,
Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary
```

The extracted CSV is approximately 5.3 GB and is intentionally not tracked by ordinary git. The compressed archive is approximately 1.4 GB and is tracked with Git LFS, so users cloning this repository should install Git LFS before pulling the dataset archive.
