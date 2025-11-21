from sklearn.metrics import silhouette_score
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from sklearn.preprocessing import normalize
from matplotlib import colors as mcolors


COLORS = [np.array(x)/255. for x in [(31, 119, 180), (255, 127, 14), (44, 160, 44),
    (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194),
    (127, 127, 127), (188, 189, 34), (23, 190, 207), (174, 199, 232),
    (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
    (196, 156, 148), (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229)]]

def plot_network(G, quantities, pos=None, layout_engine="neato", nodesize=None, suffix='',
        do_plot_labels=True, bicolor=False, add_cbar=True, cbar_label=None, label_size=7.5, highlights=None, legend_dict=None,
        do_return_pos=False, vmin=None, vmax=None, vave=None, bicolormap=None):

    nodelist = G.nodes()
    edgelist = G.edges()

    if len(nodelist)>1000:
        layout_engine="sfdp"

    if nodesize==None:
        nodesize = 200
    elif type(nodesize) == dict:
        nodesize = [100*nodesize[i] for i in nodelist]

    if pos == None:
        #pos = nx.spring_layout(G, iterations=1000, k=3*(1./np.sqrt(len(G))), threshold=0.00001)
        pos = nx.nx_agraph.graphviz_layout(G, prog=layout_engine, args='-Goverlap=false')
        #pos = nx.spring_layout(G, k=1./np.sqrt(G.order()), seed=1)


        #spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None)

    labels = nx.get_node_attributes(G, "label")
    for quantity in quantities:
        quantity_name = quantity["name"]
        quantity_values = quantity["values"]
        if cbar_label==None:
            cbar_label = quantity_name
        
        fig, ax = plt.subplots(figsize=(20,20), facecolor=(1, 1, 1))


        if bicolor:
            cmap = plt.cm.bwr
        else:
            cmap = plt.cm.Reds
        if bicolormap!=None:
            cmap = bicolormap


        if quantity["discrete"]:
            #d_color = {j:i for i,j in enumerate(sorted(set(list(quantity_values.values()))))}
            #colorlist = [COLORS[d_color[quantity_values[i]]%len(COLORS)] for i in nodelist]
            if highlights is not None:
                colorlist = [COLORS[quantity_values[i]%len(COLORS)].tolist()+[0.7] for i in nodelist]
            else:
                colorlist = [COLORS[quantity_values[i]%len(COLORS)] for i in nodelist]
            nx.draw_networkx_nodes(G, pos,
                nodelist=nodelist,
                node_size=nodesize,
                node_color=colorlist,
                edgecolors='k',
                linewidths=.5)

            if highlights is not None:
                nx.draw_networkx_nodes(G, pos,
                    nodelist=[x for x in nodelist if x in highlights],
                    node_size=[nodesize[x_i] for x_i,x in enumerate(nodelist) if x in highlights],
                    node_color=[colorlist[x_i][:-1]+[1.] for x_i,x in enumerate(nodelist) if x in highlights],
                    edgecolors='r',
                    linewidths=2.)
            
            if legend_dict is not None:
                for i,j in legend_dict.items():
                    if highlights is not None:
                        ax.scatter([0],[0],label=i, s=300, color=COLORS[j%len(COLORS)], alpha=0.7)
                    else:
                        ax.scatter([0],[0],label=i, s=300, color=COLORS[j%len(COLORS)], alpha=1)
                ax.scatter([0],[0], s=300, color="white", alpha=1)
                
                ax.legend(loc=0, fontsize=20)


        else:
            if vmin==None:
                vmin = np.min(list(quantity_values.values()))
            if vmax==None:
                vmax = np.max(list(quantity_values.values()))
            if vave==None:
                vave = np.mean(list(quantity_values.values()))

            print(vmin, vave, vmax)
            norm_ = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vave, vmax=vmax)

            drawn_nodes = nx.draw_networkx_nodes(G, pos,
                nodelist=nodelist,
                node_size=nodesize,
                #node_color=[norm_(quantity_values[i]) for i in nodelist], DOES NOT SEEM RIGHT
                node_color=[quantity_values[i] for i in nodelist],
                edgecolors='k',
                linewidths=.5,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax)

            if highlights is not None:
                nx.draw_networkx_nodes(G, pos,
                    nodelist=[x for x in nodelist if x in highlights],
                    node_size=[nodesize[x_i] for x_i,x in enumerate(nodelist) if x in highlights],
                    #node_color=[colorlist[x_i] for x_i,x in enumerate(nodelist) if x in highlights],
                    node_color='none',#[quantity_values[x_i] for x_i,x in enumerate(nodelist) if x in highlights],
                    edgecolors='green',
                    linewidths=5.)


        nx.draw_networkx_edges(G.to_directed(), pos,
            edgelist=edgelist, alpha=0.3, width=1, connectionstyle="arc3,rad=.15", arrowstyle='-')
        
        if do_plot_labels:
            if type(label_size) != dict:
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=label_size)
            else:
                for node, (x, y) in pos.items():
                    if not (highlights is None):
                        if (node in highlights):

                            plt.text(x, y, labels[node], fontsize=label_size, color="k", ha='center', va='center')
                    else:
                        plt.text(x, y, labels[node], fontsize=label_size[node], ha='center', va='center')


        if (quantity["discrete"]==False) and add_cbar:
            
            #sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vave, vmax=vmax))
            #try:
            #sm = plt.cm.ScalarMappable(cmap=cmap, norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
            #except:
            #sm = plt.cm.ScalarMappable(cmap=cmap, norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vave, vmax=vmax))
            
            sm.set_array([])
            cbar = plt.colorbar(sm, shrink=0.5, ax=ax)#, fraction=0.047, pad=0.04)
            cbar.ax.tick_params(labelsize=20)
            cbar.ax.set_ylabel(cbar_label, rotation=90, fontdict={'fontsize': 20}, labelpad=0)


            #im_ratio = drawn_nodes.shape[0]/drawn_nodes.shape[1]
            '''
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            cbar = plt.colorbar(sm, cax=cax)#, fraction=0.047, pad=0.04)
            cax_ylim = cbar.ax.get_ylim()
            print(cax_ylim)
            ticks=np.linspace(cax_ylim[0], cax_ylim[1], 11, endpoint=True).tolist()
            print(ticks)
            #cbar.ax.set_yticks(ticks)
            #cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in ticks])
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticks)
            '''
        plt.axis('off')

        #ax.set_title(quantity_name, fontdict={'fontsize': 42})
        plt.tight_layout()
        plt.savefig('network_%s_%s' % (suffix, quantity_name[:50]), dpi=300)
        plt.close()
        if do_return_pos:
            return pos
"""
def make_SVD(M, do_standardize=True, k=None):

    print(M.shape)
    M_standardized = (M - np.mean(M, axis=0))/np.std(M, axis=0)
    
    U, S, VT = np.linalg.svd(M_standardized, full_matrices=True)
    
    reduced_info = np.cumsum(S**2)/np.sum(S**2)
    if k is None:
        k = 1+(reduced_info>0.95).tolist().index(True)
    print(k, reduced_info[k-1])
    
    fig, ax = plt.subplots(1)
    ax.plot(1+np.arange(len(reduced_info)), reduced_info)
    ax.axvline(k, color='red', linestyle='--')
    #ax.set_xlim([0,100])
    plt.grid(True)
    
    M_transf = U[:,:k].dot(np.diag(S[:k]))
    PCs = VT[:k, :].T
    
    return M_transf, PCs

def make_sim(M, alpha=0.5, do_plot=True, labels_=None):
    
    cos_dist = pairwise_distances(M, M, metric='cosine')
    euclid_dist = pairwise_distances(M, M, metric='euclidean')
    
    cos_sim = 1. - (cos_dist/2.)
    euclid_sim = 1. - (euclid_dist / np.max(euclid_dist))
    
    sim_w = .5
    A_sim = sim_w * cos_sim + (1.-sim_w) * euclid_sim
    np.fill_diagonal(A_sim, 0.)

    if do_plot:
        fig, ax = plt.subplots(figsize=(5,5))

        hm = ax.imshow(A_sim, cmap='viridis', interpolation=None)
        
        ax.set_xticks(range(A_sim.shape[1]))
        ax.set_yticks(range(A_sim.shape[0]))
        
        if labels_ is not None:
            ax.set_xticklabels(labels_, rotation=90)
            ax.set_yticklabels(labels_)
        
        plt.colorbar(hm)
        
        plt.show()
    print(A_sim.shape)
    return A_sim
"""

def make_community(G, method="louvain", return_inv=True, do_sort=True):
    """
    Generate cluster communities according to the louvain algorithm
    
    PARAMS:
        G (networkx graph): undirected networkx format graph

    RETURNS:
        communities_names (dict): dictionary of node names and assigned cluster
    """
    if method =='louvain':
        communities_ = nx.community.louvain_communities(G, seed=42, weight=None)
        modularity_best_partition = nx.community.modularity(G, communities_)
        coverage_best_partition, performance_best_partition = nx.community.partition_quality(G, communities_)

        communities = {}
        for i,x in enumerate(communities_):
            for y in x:
                communities[y] = i
    '''
    elif method == 'infomap':
        communities = {}
        im = Infomap(silent=True, two_level=True, num_trials=20, flow_model="undirected")
        mapping = im.add_networkx_graph(G)
        im.run()
        for node in im.nodes:
            communities[mapping[node.node_id]] = node.module_id
        #communities_2 = im.get_modules()
        #assert(communities==communities_2)
        modularity_best_partition = im.codelength
    '''

    if do_sort:
        communities_inv = {}
        for i,j in communities.items():
            communities_inv.setdefault(j, []).append(i)

        communities_list = list(communities_inv.values())
        communities_inv = {i:communities_list[y] for i,y in enumerate(np.argsort([len(x) for x in communities_list])[::-1])}
        
        communities = {}
        for i,j in communities_inv.items():
            for k in j:
                communities[k] = i

        if return_inv:
            return modularity_best_partition, coverage_best_partition, performance_best_partition, communities, communities_inv

        else:
            return modularity_best_partition, coverage_best_partition, performance_best_partition, communities
    else:
        if return_inv:
            communities_inv = {}
            for i,j in communities.items():
                communities_inv.setdefault(j, []).append(i)
            return modularity_best_partition, communities, communities_inv
        else:
            return modularity_best_partition, coverage_best_partition, performance_best_partition, communities            

def aaa():
	return 0

def pruning_investigation(Adj, percentiles_ = np.linspace(90,100,51), do_plot=True, backbone=False):

    modularities = []
    coverages = []
    performances = []
    num_nodes_GC = []
    num_ccs = []
    num_orphans = []
    edges_weight = []
    num_communities = []
    num_communities_nonOrphans = []
    initial_edges_weight = np.sum(Adj)
    for i,percentile_ in enumerate(percentiles_):
        #print(i, percentile_)
        Adj_pruned = copy(Adj)
        if backbone:
            Adj_pruned = make_backbone_network(Adj_pruned, alpha=percentile_)
        else:
            perc_thresh_sites = np.percentile(Adj.flatten(),percentile_)
            Adj_pruned[Adj_pruned<perc_thresh_sites] = 0.
            #Adj_pruned = sparse.csr_array(Adj_pruned)
        Gi = nx.from_numpy_array(Adj_pruned, create_using=nx.Graph())
        #Gi = nx.from_scipy_sparse_array(Adj_pruned, create_using=nx.Graph())

        Ccs = nx.connected_components(Gi)
        Ccs_sorted = sorted(Ccs, key=len, reverse=True)
        Num_ccs_nonOrphans = len([y for y in [len(x)for x in Ccs_sorted] if y>1])
        num_orphans.append((len(Ccs_sorted) - Num_ccs_nonOrphans)/len(Gi.nodes()))
        num_ccs.append(Num_ccs_nonOrphans/len(Gi.nodes()))
        Gcc = Ccs_sorted[0]
        num_nodes_GC.append(len(Gcc)/len(Gi.nodes()))
        edges_weight.append(Adj_pruned.sum()/initial_edges_weight)

        #print("GC:", num_nodes_GC[-1])
        try:
            modularity, coverage, performance, communities = make_community(Gi, return_inv=False, method="louvain", do_sort=False)
        except:
            modularity = 0
            coverage = 0
            performance = 0
        labels = [communities[x] for x in sorted(communities.keys())]
        D = 1.-Adj
        np.fill_diagonal(D, 0.)
        try:
            performance = silhouette_score(D, labels, metric='precomputed')
        except: performance=0.
        modularities.append(modularity)
        coverages.append(coverage)
        performances.append(performance)
        num_communities.append(len(np.unique(list(communities.values())))/len(Gi.nodes()))
        num_communities_nonOrphans.append(num_communities[-1]-num_orphans[-1])
    print("Max.  modularity: %f" % percentiles_[np.argmax(modularities)])
    print("Max.  silhouette: %f" % percentiles_[np.argmax(performances)])
    print("95%% GC: %f" % percentiles_[np.argmin(np.abs(np.array(num_nodes_GC)-0.95))])
    idx_percThres = np.argmax(np.diff(num_nodes_GC)/np.diff(percentiles_))
    print("Perc. threshold: %f" % np.mean(percentiles_[[idx_percThres, idx_percThres+1]]))
    
    mod_slope = np.abs([np.polyfit(percentiles_[i:i+9], modularities[i:i+9], 1)[0] for i in range(len(percentiles_)-9)])
    print(mod_slope)
    print(np.argmin(mod_slope)+4)


    print("Top Modularity Plateau:", percentiles_[np.argmin(mod_slope)+4])

    
    
    if do_plot:
        fig, ax = plt.subplots(2,2, figsize=(10,10))
        
        ax[0][0].plot(percentiles_, modularities, '-o',label='modularity')
        ax[0][0].plot(percentiles_, coverages, '-o',label='coverage')
        ax[0][0].plot(percentiles_, performances, '-o',label='silhouette')
        ax[0][0].plot(percentiles_, num_nodes_GC, '-o',label='GC')
        ax[0][0].plot(percentiles_, num_ccs, '-o',label='Num CCs')
        ax[0][0].plot(percentiles_, edges_weight, '-o',label='Edges Weight')        
        ax[0][0].plot(percentiles_, num_communities, '-o',label='num_communities')
        #ax[0].plot(percentiles_, num_communities_nonOrphans, '-o',label='num_communities non orphans')
        ax[0][0].minorticks_on()
        if backbone:
            ax[0][0].set_xscale("log")
        ax[0][0].grid(True, axis='x',which='both')
        ax[0][0].legend()
        
        #ax[1].plot(num_nodes_GC, modularities, '-o', label='modularity')
        #ax[1].plot(num_nodes_GC, coverages, '-o',label='coverage')
        ax[0][1].plot(num_orphans, performances, '-o',label='silhouette (orphans)')
        ax[0][1].plot(num_orphans, num_communities, '-o',label='num_communities')
        ax[0][1].plot(num_orphans, num_communities_nonOrphans, '-o',label='num_communities non orphans')
        ax[0][1].plot(num_orphans, modularities, '-o', label='modularity (orphans)')
        #ax[1].plot(num_orphans, edges_weight, '-o', label='Edges Weight (orphans)')
        ax[0][1].legend()

        ax[1][0].plot(num_communities, modularities, '-o',label='modularity')
        ax[1][0].plot(num_communities, performances, '-o',label='silhouette')
        ax[1][0].legend()

        ax[1][1].plot(num_communities_nonOrphans, modularities, '-o',label='modularity')
        ax[1][1].plot(num_communities_nonOrphans, performances, '-o',label='silhouette')
        ax[1][1].legend()

        
        for i in range(len(percentiles_)):
            #ax[1].text(num_nodes_GC[i], modularities[i], "%.4f"%percentiles_[i])
            #ax[1].text(num_nodes_GC[i], coverages[i], "%.4f"%percentiles_[i])
            ax[1][0].text(num_communities[i], modularities[i], "%.4f"%percentiles_[i])
            ax[1][0].text(num_communities[i], performances[i], "%.4f"%percentiles_[i])
            ax[1][1].text(num_communities_nonOrphans[i], modularities[i], "%.4f"%percentiles_[i])
            ax[1][1].text(num_communities_nonOrphans[i], performances[i], "%.4f"%percentiles_[i])
            #ax[1].text(num_nodes_GC[i], num_ccs[i], "%.4f"%percentiles_[i])
            #ax[1].text(num_orphans[i], edges_weight[i], "%.4f"%percentiles_[i])
    
        
        plt.show()




def make_network(Adj, t=95., labels=None, do_plot=True, suffix="cutoff"):
    perc_thresh = np.percentile(Adj.flatten(),t)
    Adj_pruned = copy(Adj)
    Adj_pruned[Adj_pruned<perc_thresh] = 0.
    G = nx.from_numpy_array(Adj_pruned, create_using=nx.Graph())
    if labels is None:
        labels = np.arange(len(G.nodes()))
    nx.set_node_attributes(G, dict(enumerate(labels)), "label")
    print(len(G.nodes()), len(G.edges()), len(G.edges())/len(G.nodes()))
    pos = nx.spring_layout(G, k=1./np.sqrt(G.order()), seed=1)
    #pos = nx.nx_agraph.graphviz_layout(G, prog="neato", args='-Goverlap=false')
    _,_,_, communities_attr, communities_attr_to_nodes_dict = make_community(G, return_inv=True)
    nx.set_node_attributes(G, communities_attr, 'part')
    
    if do_plot:
        quantities = [{"name":"communities", "values":communities_attr, "discrete":True}]
        plot_network(G, quantities, nodesize=600, suffix=suffix, label_size=14)
    
    return G, communities_attr, communities_attr_to_nodes_dict

def make_backbone_network(M, labels=None, alpha=0.05, reciprocated=False, return_networkx=False, do_plot=True, suffix="backbone"):
    """
    Generates the backbone of a network
    
    PARAMS:
        M (numpy array): adjacency matrix
    
    OPTIONAL:
        labels (list): list of objects sorted as in M to be used as labels for nodes.
                Used only if networkx is returned. default: []
        alpha (float): disparity filter tuning the backbone extraction. Ranges from 0 to 1.
                1 returns the complete graph, 0 returns no edges. Default: 0.05
        reciprocated (bool): retains one only if in the backbone of both nodes.
                Default: False.
        return_networkx (bool): Returns a NetworkX object if True.
                Otherwise the backbone adjacency matrix. Default: False
    
    RETURNS:
        M_alpha (numpy array): Backbone adjacency matrix.
        G (networkx graph): Backbone network as NetworkX undirected graph
    
    USAGE EXAMPLE:
        M_backbone = make_backbone_network(M, alpha=0.01)

    """
    np.fill_diagonal(M, 0.)
    n = M.shape[0]
    k = np.tile(np.sum(M.astype(bool), axis=1).reshape(n, 1), (1,n))
    M_norm = normalize(M, axis=1, norm='l1')

    M_alpha = copy(M)

    idx_to_zero = (1-M_norm)**(k-1)>=alpha
    if reciprocated:
        idx_to_zero = idx_to_zero | idx_to_zero.T
    else:
        idx_to_zero = idx_to_zero & idx_to_zero.T
    M_alpha[idx_to_zero]=0.

    if not return_networkx:
        return M_alpha

    G = nx.from_numpy_array(M_alpha)
    if labels is None:
        labels = np.arange(len(G.nodes()))
    nx.set_node_attributes(G, dict(enumerate(labels)), "label")
    print(len(G.nodes()), len(G.edges()), len(G.edges())/len(G.nodes()))
    pos = nx.spring_layout(G, k=1./np.sqrt(G.order()), seed=1)
    #pos = nx.nx_agraph.graphviz_layout(G, prog="neato", args='-Goverlap=false')
    _,_,_, communities_attr, communities_attr_to_nodes_dict = make_community(G, return_inv=True)
    nx.set_node_attributes(G, communities_attr, 'part')
    
    if do_plot:
        quantities = [{"name":"communities", "values":communities_attr, "discrete":True}]
        plot_network(G, quantities, nodesize=600, suffix=suffix, label_size=14)
    
    return G, communities_attr, communities_attr_to_nodes_dict


def make_topNode_perCommunity(M_communities, node_score, feature_names, do_sort=True, do_ave=False, size_communities=None):
    if not do_sort:
        community_score = -np.arange(M_communities.shape[0])
    else:
        if do_ave:
            community_score = (M_communities.dot(node_score))/size_communities
        else:
            community_score = M_communities.dot(node_score)
    sorted_traits_idx = []
    for i in np.argsort(community_score)[::-1]:
        node_score_inCommunity = copy(node_score)
        node_score_inCommunity[~(M_communities[i].astype(bool))] = -np.inf
        sorted_traits_idx.append(np.argmax(node_score_inCommunity))
        #print(i, feature_names[sorted_traits_idx[-1]])
    return sorted_traits_idx

def get_top_nodes(G, communities_attr_to_nodes_dict, metric=None, cluster_coverage=0.95):
    
    l_flatten = []
    l_all = []
    for i,j in communities_attr_to_nodes_dict.items():
        G_cluster_nodeview = G.subgraph(j)
        G_cluster_nodelist = list(G_cluster_nodeview)
        G_cluster = nx.Graph(G_cluster_nodeview)
        dict_node_idx = {v:u for u,v in dict(enumerate(G_cluster_nodelist)).items()}
        #print(i, G_cluster_nodelist)
        c=0
        l = []
        num_nodes = G_cluster.number_of_nodes()
        ADJ_inCluster = nx.adjacency_matrix(G_cluster).todense()
        #print(ADJ_inCluster.shape)
        np.fill_diagonal(ADJ_inCluster, 1.)
        if metric is not None:
            metric_inCluster = metric[G_cluster_nodelist].reshape(-1,1)
            metric_inCluster_sum = np.sum(metric_inCluster)
        while c<cluster_coverage:
            if metric is not None:
                top_degree_idx = np.argmax(ADJ_inCluster.dot(metric_inCluster))
                top_degree_node = G_cluster_nodelist[top_degree_idx] 
            else:
                G_degree = sorted(G_cluster.degree(), key=lambda item: item[1], reverse=True)
                top_degree_node = G_degree[0][0]
            l.append(top_degree_node)
            neigh_top_degree = list(G_cluster.neighbors(top_degree_node))
            if metric is not None:
                for u in [top_degree_node]+neigh_top_degree:
                    c+=metric_inCluster[dict_node_idx[u]]/metric_inCluster_sum
                    metric_inCluster[dict_node_idx[u]] =0.
            else:
                G_cluster.remove_nodes_from([top_degree_node]+neigh_top_degree)
                c+=(len(neigh_top_degree)+1)/num_nodes
        l_flatten.extend(l)
        l_all.append(l)

    return l_flatten, l_all


