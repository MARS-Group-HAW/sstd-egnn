## Evaluation model EGGN for SSTD.

### Dependencies

- python >= 3.8
- torch >= 1.6.0
- dgl == 0.6.1
- numpy >= 1.21.5
- pandas >= 1.4.2

### Preprocessing for interaction datasets

Before running the EGNN model, you need to preprocess the downloaded public data to generate two types of interaction
datasets: behavior and mobility for placements. Follow the steps below:

1. Preprocess the raw behavior data:

```
python behavior.py --raw-in RAW_BV --out SELECTED_OUTPUT
```

2. Preprocess the raw mobility data:

```
python mobility.py --raw-in RAW_MB --out SELECTED_OUTPUT
```

### How to run

The model is first pre-trained with our self-supervised learning:

```
python essl.py --ds hamburg --locality 0.1
```

Then the model loads the pre-trained file and performs the inference:

```
python infer.py --ds hamburg --trained-model SELECTED_OUTPUT
```

Feel free to modify the parameters according to your needs and dataset.