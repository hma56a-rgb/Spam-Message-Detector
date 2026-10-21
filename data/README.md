# Data

`sms.tsv` is the SMS Spam Collection: 5,572 English text messages labelled `ham` or `spam`, tab-separated with no header (`label<TAB>message`).

- Original source: UCI Machine Learning Repository, *SMS Spam Collection*
- Authors: Tiago A. Almeida, José María Gómez Hidalgo, Akebo Yamakami (2011), "Contributions to the study of SMS spam filtering: new collection and results", ACM DocEng.
- License: CC BY 4.0
- This copy was taken from a public GitHub mirror (`justmarkham/pycon-2016-tutorial`) for convenience.

403 of the messages are exact duplicates; they are removed before splitting (see `notebooks/02_model_training.ipynb`).
