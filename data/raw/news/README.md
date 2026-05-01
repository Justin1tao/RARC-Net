# Raw Financial News Corpus

Place the raw financial-news corpus at:

```text
data/raw/news/All_external.csv
```

The local release file used in this project has approximately 13,057,514 rows and the following columns:

```text
Date, Article_title, Stock_symbol, Url, Publisher, Author,
Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary
```

The file is approximately 5.3 GB and is intentionally not tracked by ordinary git because it exceeds GitHub's normal file-size limits. For public replication, distribute this file through Git LFS with sufficient quota or through an external data artifact service, then keep the same local path before running the FinBERT conversion scripts.
