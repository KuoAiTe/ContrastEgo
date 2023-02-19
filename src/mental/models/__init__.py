from .mlp.mlp import MLPWrapper

from .gnn.gnn import GCNWrapper, GraphSAGEWrapper, GATWrapper
from .mentalnet.mentalnet import MentalNet, MentalNet_Original, MentalNet_SAGE, MentalNet_GAT, MentalNet_GAT2, MentalNetNaive
from .mentalnetdysat.mentalnetdysat import MentalNetDySAT, MentalNetDySAT_SimSiam
#from .mentalplus.mentalplus import MentalPlus, MentalPlus_TorchAttention, MentalPlus_CON, MentalPlus_CON_BASE, MentalPlus_CON_GRAPH, MentalPlus_CON_USE_GNN, MentalPlus_CON_NO_HGNN, MentalPlus_CON_USE_NODE, MentalPlus_CON_USE_GRAPH, MentalPlus_CON_Without_Ttransformer, MentalPlus_CON_BASE_Without_Ttransformer, MentalPlus_CON_BASE_NO_GRAPH_AGGREGATION, MentalPlus_CON_NO_GRAPH_AGGREGATION
from .mentalplus.mentalplus import *
from .ugformer.ugformer import FullyConnectedGT_UGformerV2 as UGformer
from .evolvegcn.evolvegcn import EvolveGCN