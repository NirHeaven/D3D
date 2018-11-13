# D3D
## Introduction   

This respository is implementation of the proposed method in [LRW-1000: A Naturally-Distributed Large-Scale Benchmark for Lip Reading in the Wild](). Our paper can be found [here](https://arxiv.org/pdf/1810.06990.pdf).
## Dependencies
* python 3.6.7   
* pytorch 1.0.0.dev20181103
* scipy 1.1.0
## Dataset
This model is pretrained on LRW with RGB lip images(112×112), and then tranfer to LRW-1000 with the same size.    
## Training   
You can train the model as follow:
```
python main.py --data_root "data path" --index_root "index root"
```
Where the `data_root` and `index_root` specifys the "LRW-1000 data path" and "label path" correspondly.   
All the parameters we use is set as default value in [args.py]().You can also pass parameters through console just like:
```
python main.py --gpus 0,1 --batch_size XXX --lr 1e-4 --data_root "data path" --index_root "index root"
```
**Note**:   
Please pay attention to that you may need modify the code in [dataset.py]() and change the parameters `data_root` and `index_root` to make the scripts work just as expected. 
## Reference

If this repository was useful for your research, please cite our work:

```
@article{petridis2018end,
  title={LRW-1000: A Naturally-Distributed Large-Scale Benchmark for Lip Reading in the Wild},
  author=Shuang Yang, Yuanhang Zhang, Dalu Feng, Mingmin Yang, Chenhao Wang, Jingyun Xiao, Keyu Long, Shiguang Shan, Xilin Chen},
  booktitle={arXiv},
}
```
