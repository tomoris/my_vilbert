# my_vilbert

This is an extension ViLBERT for various types of non textual data.
We use vilbert_beta repository.
<https://github.com/jiasenlu/vilbert_beta>

## Prerequisites

```console
cd src/model
wget https://raw.githubusercontent.com/jiasenlu/vilbert_beta/master/vilbert/vilbert.py
wget https://raw.githubusercontent.com/jiasenlu/vilbert_beta/master/vilbert/utils.py

cd ../../
poetry install
poetry shell
```

## Usage

pretrain a model on sample text file.

make features of non-textual data.

```python
import numpy as np

non_text_size = 7
feat_size = 100
non_text_feat_array = np.random.rand(non_text_size, feat_size)
np.save("sample/pretrain_sample.non_text_feat.npy", non_text_feat_array)
```

execute pretraining and visualize info.

```console
./exp.sh
python src/visualization/visualize_learning_info.py
```
