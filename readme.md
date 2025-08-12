### Environment
```
conda create -n llactr python=3.10.10
conda activate llactr
pip install -r requirements.txt
```

### Dataset preprocess
1. Amazon-2023
- download: [link](https://amazon-reviews-2023.github.io/)
```
cd LLaCTR/data/gift
gunzip meta_Gift_Cards.jsonl.gz
gunzip Gift_Cards.jsonl.gz
python process.py

cd LLaCTR/data/game
gunzip meta_Video_Games.jsonl.gz
gunzip Video_Games.jsonl.gz
python process.py

cd LLaCTR/data/music
gunzip meta_Digital_Music.jsonl.gz
gunzip Digital_Music.jsonl.gz
python process.py
```

2. MovieLens-1M
- download: [link](https://grouplens.org/datasets/MovieLens/1m/)
```
cd LLaCTR/data/ml1m
unzip ml-1m.zip
python process.py
```

### Run backbone
```
cd LLaCTR/ctr

- run
python run_expid.py --config ./config_01 --expid FM_gift --adding_mode 0 --weight_decay 0.01  --nlp_field 13 --gpu 0

- tune
nnictl create --config config_gift.yml -p 8100
```

### Run LLaCTR
```
cd LLaCTR/data/gift

python generate_instruction.py

cd LLaCTR/llm

bash train_emb.sh
bash encode.sh

cd LLaCTR/ctr

- run
python run_expid.py --config ./config_01 --expid FM_gift --adding_mode 1 --nlp_field 13  --llm_emb_path2 "../llm/llm_emb/gift/sl_1000/field_sl_eos_epoch1.pt" --lambda_loss 0.5 --lambda_llm 0.01 --weight_decay 0.01 --gpu 0

- tune
nnictl create --config config_FM_gift_llm.yml -p 8101
```

