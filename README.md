<div align="center" style="font-family: charter;">

<h1><i>VIR-Bench</i>:</br> Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction</h1>

<img src="icons/top.png" width="95%"/>
<br />

<a href="FIXME" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-traveling--across--languages-red?logo=arxiv" height="20" />
</a>

<div>
    <a href="https://www.conan1024hao.com/" target="_blank">Hao Wang*</a><sup>1</sup>,
    <a href="https://www.linkedin.com/in/eiki-murata/" target="_blank">Eiki Murata*</a><sup>2,3</sup>,
    <span>Lingfang Zhang<sup>1</sup>,</span>
    <span>Ayako Sato<sup>2</sup>,</span>
    <span>So Fukuda<sup>1</sup>,</span>
    <span>Ziqi Yin<sup>1</sup>,</span>
    <span>Wentao Hu<sup>1</sup>,</span>
    <span>Keisuke Nakao<sup>1</sup>,</span>
    <span>Yusuke Nakamura<sup>1</sup>,</span>
    <span>Sebastian Zwirner<sup>1</sup>,</span>
    <span>Yi-Chia Chen<sup>1</sup>,</span>
    <span>Hiroyuki Otomo<sup>2</sup>,</span>
    <span>Hiroki Ouchi<sup>4,2</sup>,</span>
    <span>Daisuke Kawahara<sup>1</sup></span>
</div>
<br />

<div>
    <sup>1</sup>Waseda University&emsp;
    <sup>2</sup>CyberAgent, Inc.&emsp;
    <sup>3</sup>AI Shift, Inc.&emsp;
    <sup>4</sup>Nara Institute of Science and Technology&emsp;
</div>

<div>
    * Equal contribution
</div>
<br />

<p align="justify"><i>Recent advances in multimodal large language models (MLLMs) have significantly enhanced video understanding capabilities, opening new possibilities for practical applications. Yet current video benchmarks focus largely on indoor scenes or short-range outdoor activities, leaving the challenges associated with long-distance travel largely unexplored. Mastering extended geospatial-temporal trajectories is critical for next-generation MLLMs, underpinning real-world tasks such as embodied-AI planning and navigation. To bridge this gap, we present VIR-Bench, a novel benchmark consisting of 200 travel videos that frames itinerary reconstruction as a challenging task designed to evaluate and push forward MLLMs' geospatial-temporal intelligence. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, struggle to achieve high scores, underscoring the difficulty of handling videos that span extended spatial and temporal scales. Moreover, we conduct an in-depth case study in which we develop a prototype travel-planning agent that leverages the insights gained from VIR-Bench. The agent’s markedly improved itinerary recommendations verify that our evaluation protocol not only benchmarks models effectively but also translates into concrete performance gains in user-facing applications.</i></p>

</div>

## Release
- `2025-09-20` :rocket: We released the benchmark together with its evaluation framework and agent implementations.

## Contents
- [Release](#release)
- [Contents](#contents)
- [VIR-Bench](#vir-bench)
- [Experiments](#experiments)
  - [Task Definition](#task-definition)
  - [Results](#results)
- [Download the Dataset](#download-the-dataset)
- [RUN Your Own Evaluation](#run-your-own-evaluation)
  - [Installation](#installation)
  - [Evaluation](#evaluation)
- [Travel-planning Agent](#travel-planning-agent)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## VIR-Bench

## Experiments
### Task Definition
### Results

## Download the Dataset
We release the VIR-Bench dataset strictly for research purposes, in compliance with <ins>Article 30-4 (Use for Non-Enjoyment Purposes)</ins> and <ins>Article 47-5 (Minor Use in Information Analysis Services)</ins> of the Japanese Copyright Act. Commercial use of any kind is strictly prohibited. The dataset may not be redistributed on servers outside Japan or under alternative licenses.

Dataset link: https://soya.infini-cloud.net/share/1302266998c5d047

Access phrase: `waseda`

To run evaluations, download and unzip `data.zip` and `videos.zip`, and organize them into the following directory structure.
`graphs.zip` (containing visiting order graphs in both pickle and SVG formats) is optional and not required for evaluation.
```
VIR-Bench/
  ├── data/
  │   ├── test-00000-of-00001.parquet
  │   └── validation-00000-of-00001.parquet
  ├── videos/
  │   ├── 0oODCXC3oms.mp4
  │   └── ...
```

## RUN Your Own Evaluation
### Installation

### Evaluation

## Travel-planning Agent
We provide the full code for the travel-planning agent used in our paper. See the [agent/README](https://github.com/nlp-waseda/VIR-Bench/blob/main/agent/README.md) for setup and usage instructions.

## Acknowledgement
The evaluatiokn code is build upon [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We acknowledge their team for providing this excellent toolkit for evaluating multimodal large language models.

## Citation
FIXME
