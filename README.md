# POFHP
Code for AAAI 2025 paper --- **P**ublic **O**pinion **F**ield Effect and **H**awkes **P**rocess Join Hands for Information Popularity Prediction.

# Overview
```bash
POFHP:.
│  get_data.py
│  pytorchtools.py
│  requirements.txt
│          
├─data
│  ├─android
│  │      
│  ├─christianity
│  │
│  ├─douban
│  │      
│  └─twitter
│          
└─src
    │  main.py
    │  models.py
    │  util.py
    │  
    └─ckpt
```
1. **get_data.py**: This file is used to process the data.
2. **pytorchtools.py**: This file is used to define the early_stopping mechanism.
3. **requirements.txt**: Dependencies file.
4. **data/**：Dataset folder.
5. **src/main.py**: Main file.
6. **src/models.py**: POFHP implementation.
7. **src/util.py**: Defining various toolkits.

# Dependencies
Please install the following packages:
```
gensim==3.8.3
joblib==1.3.2
matplotlib==3.7.5
networkx==3.1
numpy==1.24.4
pandas==2.0.3
scikit-learn==1.3.2
scipy==1.10.1
torch==2.1.2+cu121
torch-cluster==1.6.3+pt21cu121
torch-geometric==2.5.2
torch-scatter==2.1.2+pt21cu121
torch-sparse==0.6.18+pt21cu121
torch-spline-conv==1.2.2+pt21cu121
tqdm==4.66.2
transformers==4.39.0
```
You can also simply run:
```
pip install -r requirements.txt
```
# Usage
Since the Android and Twitter datasets are large, we compressed them. Therefore, the two datasets should be decompressed before use:
```bash
cd data/
unzip douban.zip
unzip twitter.zip
```
Then:
```bash
cd src/
python main.py --data_name christianity
python main.py --data_name android
python main.py --data_name twitter
python main.py --data_name douban
```

# Cite
```bash
@inproceedings{li2025public,
  title={Public Opinion Field Effect and Hawkes Process Join Hands for Information Popularity Prediction},
  author={Li, Junliang and Yang, Yajun and Zhang, Yujia and Hu, Qinghua and Zhao, Alan and Gao, Hong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={11},
  pages={12076--12083},
  year={2025}
}
```