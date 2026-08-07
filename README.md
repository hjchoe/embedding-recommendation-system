# Embedding Recommendation System

Experimental evaluation of lightweight related-video recommendation methods for the [Centaur Learning](https://github.com/centaurinstitute/learning.centaurinstitute.org) catalog.

## Status

Research prototype only. This repository is not connected to the production learning site and does not use viewer history, personal information, production credentials, or production services.

## Objective

Determine whether semantic title embeddings, curated video tags, or a hybrid ranking method can produce useful related-video results across the current learning catalog.

The initial target is item-to-item related-video ranking, not personalized user recommendations.

## Planned comparisons

1. Current learning-site behavior as the baseline.
2. Title embedding cosine similarity.
3. Title and humanized tags embedded together.
4. Title embedding similarity combined with weighted exact-tag overlap.
5. Title plus boilerplate description as a negative control.

Results will be evaluated manually across tagged and untagged videos, Summer Schools and Winter Workshops, informative and generic titles, duplicate titles, and different presentation formats.

## Current data findings

- Titles are the strongest complete semantic signal.
- Tags contain useful topic information but are missing from much of the catalog.
- Existing descriptions are structurally populated but contain repeated generic boilerplate.
- Categories describe presentation format rather than subject matter.
- No viewer-history or behavioral recommendation data is currently used.

## Repository boundaries

This repository contains offline experiments, evaluation procedures, and design decisions. Any production integration will be proposed separately through the learning-site repository’s normal branch and pull-request workflow.

Generated embeddings, downloaded models, credentials, private data, and environment files must not be committed.

## Data provenance

Catalog metadata originates from the Apache-2.0-licensed [Centaur Institute learning-site repository](https://github.com/centaurinstitute/learning.centaurinstitute.org). Any committed snapshot will identify its source path and exact source commit.

## License

Apache License 2.0. See [LICENSE](LICENSE).
