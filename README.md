# D3D：The implementation of proposed method in [LRW-1000](https://arxiv.org/pdf/1810.06990.pdf)

-------------------------------

This model is pretrained on LRW with RGB lip images(112×112), and then tranfer to LRW-1000.   
You can train the model as follow:
```
python main.py --data_root "data path" --index_root "index root"
```
Where the [data_root]() and [index_root]() specifys the "LRW-1000 data path" and "label path" correspondly.   
All the parameters we use is set as default value in [args.py]().You can also pass parameters through console just like:
```
python main.py --gpus 0,1 --batch_size XXX --lr 1e-4 --data_root "data path" --index_root "index root"
```
Please pay attention to that you may need modify the code in [dataset.py]() and change the parameters [data_root]() and [index_root]() to make the scripts work just as expected. 
