# Paper Summary

## Problems Addressed
Video understanding datasets currently available,
* include common activities only, ignoring other topics/domains
* have answers ( labels ) in multiple-choice format, which is mostly not available in real-world settings
* contain videos of shorter durations, performance is well-studied on larger datasets.

## Contributions
* WildQA dataset
* Tasks on WildQA - 1) Video QA and 2) Evidence Selection from videos ( for better interpretability )
* Evaluation with few shot learning

## Methods
* [Text + Visual models](https://arxiv.org/pdf/2104.04182v3.pdf) -> Encoder-Decoder Architecture with Transformer model. They used I3D ( 3D ConvNets ) as video encoder and a Text2Text transformer for input queries.
* [Multi Task Learning](https://ruder.io/multi-task/)

