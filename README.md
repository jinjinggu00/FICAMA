# FICAMA
Official pytorch implementation of Skeleton-Based Few-Shot Action Recognition via Fine-Grained Information Capture and Adaptive Metric Aggregation
# Environment
pytorch = 1.13.1

python = 3.9.16

tqdm = 4.65.0

pynvml = 11.5.0

fitlog = 0.9.15
# Datasets
NTU-T, NTU-S, and Kinetics are composed of [https://github.com/NingMa-AI/DASTM](https://github.com/NingMa-AI/DASTM) Provide.

# Data Storage Format

Our data is organized in the following folder structure:

- data/
  - kinetics_2d/
  - kinetics_clean/
  - kinetics/
    - Kinetics/
  - ntu1s/
  - ntu120/
    - NTU-S/
    - NTU-T/

# Run
Run our proposed method on the NTU-T dataset with 5-way 1-shot settings:
```python
python train.py --metric en2 --mix 1 --mixjoint 1 --sat 1 --mi -0.5
```
Run our proposed method on the NTU-T dataset with 5-way 5-shot settings:
```python
python train.py --metric en2 --mix 1 --mixjoint 1 --sat 1 --mi -0.5 -nsTr 5 -nsVa 5
```
Pass in different parameters to run different datasets:
```python
--dataset    ntu-T
--dataset    ntu-S
--dataset    kinetics
--dataset    kinetics_2d
--dataset    kinetics_clean
```
Using different backbones:
```python
python train.py  --backbone ctrgcn
python train.py  --backbone stgcnpp
python train.py  --backbone hdgcn
```
## Comprehensive documentation and detailed usage instructions are in progress. The codebase is currently undergoing refinement and optimization. We appreciate your patience as we work to provide a more robust and well-documented implementation. Updates will be made available as soon as possible.

# Acknowledgements
This repository is based on [DASTM](https://github.com/NingMa-AI/DASTM).

We appreciate the original authors for their significant contributions.
