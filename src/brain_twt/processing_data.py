import networkx as nx
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import *
import pandas as pd
from tqdm import tqdm
import numpy as np
from os.path import join, exists
import pickle


class TemporalGraph():
    def __init__(self, data: pd.DataFrame, time_granularity: str, dataset_name: str):
        '''
        :param data: DataFrame- source, target, time, weight columns
        :param time_granularity: 'days', 'weeks', 'months', 'years' or 'hours'
        '''

        if 'weight' not in data.columns:
            data['weight'] = 1

        self.data = data
        self.time_granularity = time_granularity
        self.data['time_index'] = data['time']

    def get_static_graph(self):
        g = nx.from_pandas_edgelist(self.data, source='source', target='target', edge_attr='time',
                                    create_using=nx.MultiDiGraph())
        self.nodes = g.nodes()
        return g

    # def filter_nodes(self, thresh: int = 5):
    #     # nodes2filter = [node for node, degree in self.static_graph.degree() if degree < thresh]
    #     return []

    def get_temporal_graphs(self, min_degree: int, mode: str = 'dynamic') -> dict:
        '''

        :param min_degree: int.  filter nodes with degree<min_degree in all time steps
        :param mode: if not 'dynamic', add all nodes to the current time step without edges
        :return: dictionary. key- time step, value- nx.Graph
        '''
        G = {}
        print('getting temporal graphs')
        for t, time_group in tqdm(self.data.groupby('time')):
            time_group = time_group.groupby(['source', 'target'])['weight'].sum().reset_index()
            g = nx.from_pandas_edgelist(time_group, source='source', target='target', edge_attr=['weight'],
                                        create_using=nx.DiGraph())
            if mode != 'dynamic':
                g.add_nodes_from(self.nodes)
            g.remove_nodes_from(self.filter_nodes(min_degree))
            G[t] = g
        self.graphs = G
        return G


def load_dataset(graph_df: pd.DataFrame, time_granularity: str):# -> tuple[nx.Graph, TemporalGraph]:
    '''

    :param graph_df:  DataFrame- source, target, time, weight columns
    :param dataset_name: name of the dataset
    :param time_granularity: the time granularity of the graphs time steps- can be 'days', 'weeks', 'months', 'years' or 'hours'
    :return:
    '''
    temporal_g = TemporalGraph(data=graph_df, time_granularity=time_granularity, dataset_name=None)
    graph_df = temporal_g.data
    graph_df['time'] = graph_df['time_index']
    graph_nx = nx.from_pandas_edgelist(graph_df, 'source', 'target', edge_attr=['time'],
                                       create_using=nx.MultiDiGraph())

    return graph_nx, temporal_g

