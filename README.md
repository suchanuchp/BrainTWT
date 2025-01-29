# BrainTWT
### Installation ###
The code was tested on Python 3.8.4. The requirements are in `setup.py`.
```
cd brain_twt
pip install -e .
```

### Data ###
Example data format for ABIDE dataset can be found in `brain_twt/example_data`.

#### Input
The data should include - source node, target node, time of interaction, weight(optional) as csv file (graph_df).
	
	node1_id_int,node2_id_int,time_timestamp

#### Create temporal random walk sequences 
Run `create_temporal_random_walks.py` 

#### Train graph-level embedding model
Run `train_temporal_graph_model.py`