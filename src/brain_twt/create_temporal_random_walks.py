import itertools
import os
import argparse
from node2vec import Node2Vec
import numpy as np
from tqdm import tqdm
import pandas as pd
import networkx as nx
from stellargraph import StellarGraph

from processing_data import load_dataset
from random_walk import TemporalStructuralRandomWalk


def nx_to_stellar(graphs):
    edge_list = []
    for time_step, graph in graphs.items():
        for u, v in graph.edges():
            edge_list.append([u, v, time_step])
    df_graph = pd.DataFrame(edge_list, columns=['src', 'dst', 't'])
    nodes = list(set(df_graph.src).union(set(df_graph.dst)))

    dynamic_graph = StellarGraph(
        nodes=nodes,
        edges=df_graph,
        source_column='src',
        target_column='dst',
        edge_weight_column='t'
    )

    return dynamic_graph


def csv_to_stellar_graph(filepath):
    df_graph = pd.read_csv(filepath, index_col=False, names=['src', 'dst', 't'])
    df_graph.src = df_graph.src.astype(str)
    df_graph.dst = df_graph.dst.astype(str)
    nodes = list(set(df_graph.src).union(set(df_graph.dst)))
    dynamic_graph = StellarGraph(
        nodes=pd.DataFrame(index=nodes),
        edges=df_graph,
        source_column='src',
        target_column='dst',
        edge_weight_column='t',
    )
    return dynamic_graph


def create_temporal_random_walks(graphs, graph_indices, graph_labels, opt):
    walk_len = opt['maximum_walk_length']
    num_walks = opt['random_walks_per_node']
    min_walk_len = opt['min_walk_length']
    savedir = opt['savedir']
    n_nodes = opt['n_nodes']
    print(f"walk_len={walk_len}, num_walks={num_walks}")
    filename = f'paths_walk_len_{walk_len}_num_walks_{num_walks}.csv'
    savepath = os.path.join(savedir, filename)
    data_df_list = []
    avg_walk_lens = []
    for i, (graph, graph_index, graph_label) in tqdm(enumerate(zip(graphs, graph_indices, graph_labels))):
        temporal_rw = TemporalStructuralRandomWalk(graph)
        num_cw = n_nodes * num_walks * (walk_len - min_walk_len + 1)  # TODO: to factor of |T|
        walks = temporal_rw.run(
            num_cw=num_cw,
            cw_size=min_walk_len,
            max_walk_length=walk_len,
            walk_bias="exponential",
            seed=0,
            alpha=0,
            include_same_timestep_neighbors=opt['include_same_timestep_neighbors'],
        )
        len_walks = np.mean([len(walk) for walk in walks])
        avg_walk_lens.append(len_walks)
        print(f'average walk length: {len_walks}')
        walks = [[str(node) for node in walk] for walk in walks]
        sents = [" ".join(sent) for sent in walks]
        data_df_list.append(
            pd.DataFrame(np.array([sents, [graph_index] * len(sents), [graph_label] * len(sents)]).T,
                         columns=['sent', 'graph_idx', 'graph_label']))

        data_df = pd.concat(data_df_list)
        data_df.to_csv(savepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--datadir', type=str, default='data/abide')
    parser.add_argument('-s', '--savedir', type=str, default='data/abide')
    parser.add_argument('-r', '--random_walks_per_node', type=int, default=20)
    parser.add_argument('--label_path', type=str, default='data/Phenotypic_V1_0b_preprocessed1.csv')
    parser.add_argument('-l', '--maximum_walk_length', type=int, default=20)
    parser.add_argument('-w', '--min_walk_length', type=int, default=5)
    parser.add_argument('--include_same_timestep_neighbors', type=int, default=0)
    parser.add_argument('--n_nodes', type=int, default=116)

    args = parser.parse_args()
    opt = vars(args)

    data_dir = opt['datadir']
    save_dir = opt['savedir']
    label_path = opt['label_path']

    if not os.path.exists(data_dir):
        raise FileExistsError("data dir does not exist")

    if not os.path.exists(label_path):
        raise FileExistsError("label path does not exist")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_info = pd.read_csv(label_path)
    filenames = os.listdir(data_dir)
    filenames = sorted([filename for filename in filenames if filename.endswith('.csv')])

    poi_files = df_info.FILE_ID.tolist()
    filtered_filenames = []
    labels = []
    graph_indices = []
    temporal_graphs = []
    print('preprocessing')
    for filename in tqdm(filenames):
        file_id = filename[:filename.find('_func_preproc.csv')]
        if file_id in poi_files:
            filepath = os.path.join(data_dir, filename)
            filtered_filenames.append(filename)
            group = df_info[df_info.FILE_ID == file_id].iloc[0].DX_GROUP
            label = 0 if group == 2 else 1  # 0: control, 1: autism
            labels.append(label)
            graph_indices.append(file_id)
            temporal_graph = csv_to_stellar_graph(filepath)
            temporal_graphs.append(temporal_graph)

    create_temporal_random_walks(graphs=temporal_graphs, graph_indices=graph_indices, graph_labels=labels, opt=opt)


if __name__ == '__main__':
    main()
