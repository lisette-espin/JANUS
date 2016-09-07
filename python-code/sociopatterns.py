__author__ = 'espin'

######################################################################
# dependencies
######################################################################
import matplotlib
matplotlib.use('macosx')
from scipy.sparse import lil_matrix, csr_matrix
import pandas as pd
import os
import operator
import numpy as np
from scipy import io
import graph_tool.all as gt
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(); sns.set_style("whitegrid"); sns.set_style("ticks"); sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}); sns.set_style({'legend.frameon': True})

from org.gesis.libs import graph as c
from org.gesis.libs.janus import JANUS
from org.gesis.libs.graph import DataMatrix
from org.gesis.libs.hyptrails import HypTrails

######################################################################
# constants
######################################################################
FILES = ['scc2034_kilifi_all_contacts_across_households.csv','scc2034_kilifi_all_contacts_within_households.csv']
HIGH = 0.8
LOW = 0.2
LINKSONLY = False

######################################################################
# networkx
######################################################################
def loadCSV(path):
    #header: h1,m1,h2,m2,age1,age2,sex1,sex2,duration,day,hour
    df = None
    for fn in FILES:
        f = os.path.join(path,fn)
        if df is None:
            df = pd.read_csv(f, sep=',', header=0 )
        else:
            df = df.append( pd.read_csv(f, sep=',', header=0) )

    df['id1'] = df.h1.astype(str).str.cat(df.m1.astype(str))
    df['id2'] = df.h2.astype(str).str.cat(df.m2.astype(str))
    return df

def create_graph(df):
    nodes = df[['id1','h1','age1','sex1']].rename(index=str, columns={"id1":"id", 'h1':'h', "age1":"age", "sex1":"sex"})
    nodes.append( df[['id2','h2','age2','sex2']].rename(index=str, columns={"id2":"id", 'h2':'h', "age2":"age", "sex2":"sex"}) )
    nodes.drop_duplicates(inplace=True)

    G = nx.MultiGraph()
    for index, row in nodes.iterrows():
        G.add_node(row['id'], age=row['age'], sex=row['sex'], household=row['h'])

    edges = df[['id1','id2']]
    G.add_edges_from(edges.values.tolist())

    return G

def create_matrix(G):
    m = nx.adjacency_matrix(G)
    print('matrix: {}'.format(m.sum()))
    return m.tocsr()

def save_matrix(m,path,name):
    fn = os.path.join(path,name)
    io.mmwrite(fn, m)

def plot_matrix(m,path,name):
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(10,10))
    ax = sns.heatmap(m.toarray(), ax=ax,
        # annot=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"})
    ax.set_xlabel('target nodes')
    ax.set_ylabel('source nodes')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')

    plt.setp( ax.xaxis.get_majorticklabels(), rotation=90, horizontalalignment='center', fontsize=7 )
    plt.setp( ax.yaxis.get_majorticklabels(), rotation=0, horizontalalignment='center', x=1.0, fontsize=7 )

    cbar_ax.set_title('cardinality (no. of edges)')

    fn = os.path.join(path,name)
    plt.savefig(fn, dpi=1200, bbox_inches='tight')

    print('- plot adjacency done!')
    plt.close()

def plot_graph(G,path,name):
    pos = nx.spring_layout(G)
    nx.draw(G)
    fn = os.path.join(path,name)
    plt.savefig(fn, dpi=1200, bbox_inches='tight')
    plt.close()
    print(nx.info(G))
    print('Is directed: {}'.format(nx.is_directed(G)))

######################################################################
# graphtools
######################################################################
def create_graphtool(G,m):
    g = gt.Graph(directed=isdirected)
    for n1 in G.nodes():
        for n2 in G.nodes():
            source = G.nodes().index(n1)
            target = G.nodes().index(n2)

            if target >= source and G.has_edge(n1,n2):
                for i in range(m[source,target]):
                    g.add_edge(source, target, add_missing=True)

            # if G.has_edge(n1,n2):
            #     for i in range(m[source,target]):
            #         g.add_edge(source, target, add_missing=True)

    g.vp.age = vertices(G,g,'age')
    g.vp.sex = vertices(G,g,'sex')
    g.vp.household = vertices(G,g,'household')
    g.ep.multiplicity = edges_multiplicity(G,g)
    return g

def vertices(G,g,attribute):
    tmp = []
    prop = g.new_vertex_property("int")
    for n,d in G.nodes_iter(data=True):
        v = G.nodes().index(n)
        if d[attribute] not in tmp:
            tmp.append(d[attribute])
        prop[v] = tmp.index(d[attribute])
    print(attribute)
    print(tmp)
    return prop

def edges_multiplicity(G,g):
    tmp = []
    prop = g.new_edge_property("int")
    for n1,d1 in G.nodes_iter(data=True):
        for n2,d2 in G.nodes_iter(data=True):
            v1 = G.nodes().index(n1)
            v2 = G.nodes().index(n2)

            # nedges = len(g.edge(v1,v2,all_edges=True))
            # if nedges > 0:
            #     prop[(v1,v2)] = nedges

            if v2 >= v1:
                nedges = len(g.edge(v1,v2,all_edges=True))
                if nedges > 0:
                    prop[(v1,v2)] = nedges
                    if nx.is_directed(G):
                        nedges = len(g.edge(v2,v1,all_edges=True))
                    prop[(v2,v1)] = nedges
    return prop

def plot_graphtool(g,colorproperty,path,name):
    fn = os.path.join(path,name)
    gt.graph_draw(g, vertex_fill_color=colorproperty, edge_color="black", output=fn)

def plot_block_matrix(matrix,g,colorproperty,path,name):
    #### labels
    m = list(set(sorted([colorproperty[v] for v in g.vertices()]))) ### only ids of propoerties
    _labels = {i:sum([colorproperty[v]==i for v in g.vertices()]) for i in m} ### id: # of vertices
    labels = ['' for i in g.vertices()]
    counter = 0
    for b,n in _labels.items():
        index = int((n / 2) + counter)
        counter += n
        if 'household' in name:
            v=['H', 'E', 'B', 'L', 'F']
            labels[index] = '{}'.format(v[m[b]])
        else:
            labels[index] = '{}\n{}'.format(name.replace('block_','').replace('.pdf',''),m[b])

    ##### matrix sorted
    tmp = lil_matrix(matrix.shape)
    l = {v:colorproperty[v] for v in g.vertices()}
    l = sorted(l.items(), key=operator.itemgetter(1))
    row=0
    for x in l:
        v1 = x[0]
        col=0
        for y in l:
            v2 = y[0]
            # tmp[row,col] = matrix[int(v1),int(v2)]
            if v2 >= v1:
                tmp[row,col] = matrix[int(v1),int(v2)]
                tmp[col,row] = matrix[int(v2),int(v1)]
            col+=1
        row += 1

    size = (10,10)
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=size)
    ax = sns.heatmap(tmp.toarray(), ax=ax,
        #annot=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"},
                     vmax=tmp.toarray().max(),
                     xticklabels=labels,
                     yticklabels=labels
                     )
    ax.set_xlabel('target nodes')
    ax.set_ylabel('source nodes')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')
    plt.setp( ax.xaxis.get_majorticklabels(), horizontalalignment='center' )
    plt.setp( ax.yaxis.get_majorticklabels(), rotation=270, horizontalalignment='center', x=1.02 )
    cbar_ax.set_title('cardinality (no. of edges)')

    fn = os.path.join(path,name)
    plt.savefig(fn, dpi=1200, bbox_inches='tight')

    print('- plot adjacency done!')
    plt.close()
    print('matrix: {}'.format(tmp.sum()))

def plot_blockmembership_dist(g,colorproperty,path,name):
    data = {}
    title = 'Groups Histogram'
    # n = float(g.num_edges())
    for edge in g.edges():
        s = colorproperty[edge.source()]
        t = colorproperty[edge.target()]
        ### block membership
        k = '{}-{}'.format(s,t)
        if k not in data:
            data[k] = 0
        data[k] += 1 #/ n
        ### selfloops
        if edge.source() == edge.target():
            k = 'selfloop'
            if k not in data:
                data[k] = 0
            data[k] += 1 #/ n
    xx = range(len(data.keys()))
    yy = data.values()
    plt.bar(xx,yy)
    plt.xticks(xx, data.keys(), rotation='vertical')
    plt.margins(0.1)
    plt.subplots_adjust(bottom=0.1)
    plt.title(title)
    plt.ylabel("# edges")
    plt.xlabel("membership")
    fn = os.path.join(path,name)
    plt.savefig(fn, dpi=1200, bbox_inches='tight')
    plt.close()
    print('- plot block membership hist done!')

######################################################################
# hypotheses
######################################################################

def same_age(datanode1, datanode2):
    return HIGH if datanode1['age'] == datanode2['age'] else LOW

def same_sex(datanode1, datanode2):
    return HIGH if datanode1['sex'] == datanode2['sex'] else LOW

def same_household(datanode1, datanode2):
    return HIGH if datanode1['household'] == datanode2['household'] else LOW

def homophily(datanode1, datanode2):
    return similar_age(datanode1,datanode2) + same_sex(datanode1,datanode2) + same_household(datanode1, datanode2)

def different_age(datanode1, datanode2):
    return LOW if datanode1['age'] == datanode2['age'] else HIGH

def different_sex(datanode1, datanode2):
    return LOW if datanode1['sex'] == datanode2['sex'] else HIGH

def different_household(datanode1, datanode2):
    return LOW if datanode1['household'] == datanode2['household'] else HIGH

def heterophily(datanode1, datanode2):
    return different_age(datanode1,datanode2) + different_sex(datanode1,datanode2) + different_household(datanode1, datanode2)

def older(datanode1, datanode2):
    return  int( datanode2['age'] > datanode1['age'] )

def younger(datanode1, datanode2):
    return  int( datanode2['age'] < datanode1['age'])

def age_group_0(datanode1, datanode2):
    return  int( datanode1['age'] == datanode2['age'] == 0)

def age_group_1(datanode1, datanode2):
    return  int( datanode1['age'] == datanode2['age'] == 1)

def age_group_2(datanode1, datanode2):
    return  int( datanode1['age'] == datanode2['age'] == 2)

def age_group_3(datanode1, datanode2):
    return  int( datanode1['age'] == datanode2['age'] == 3)

def age_group_4(datanode1, datanode2):
    return  int( datanode1['age'] == datanode2['age'] == 4)

def difference_age_1(datanode1, datanode2):
    return int(abs( datanode1['age'] - datanode2['age'] ) == 1)

def sex_group_M(datanode1, datanode2):
    return  int( datanode1['sex'] == datanode2['sex'] == 'M')

def sex_group_F(datanode1, datanode2):
    return  int( datanode1['sex'] == datanode2['sex'] == 'F')

def household_group_F(datanode1, datanode2):
    return  int( datanode1['household'] == datanode2['household'] == 'F')

def household_group_E(datanode1, datanode2):
    return  int( datanode1['household'] == datanode2['household'] == 'E')

def household_group_B(datanode1, datanode2):
    return  int( datanode1['household'] == datanode2['household'] == 'B')

def household_group_H(datanode1, datanode2):
    return  int( datanode1['household'] == datanode2['household'] == 'H')

def similar_age(datanode1, datanode2):
    return 1 / float( 1 + abs( datanode1['age'] - datanode2['age']) )

def difference_age(datanode1, datanode2):
    return abs( datanode1['age'] - datanode2['age'] )

def same_household_and_age(datanode1, datanode2):
    return same_household(datanode1,datanode2) + same_age(datanode1,datanode2)

def build_hypothesis(G, criteriafn, selfloops=False):
    nnodes = nx.number_of_nodes(G)
    belief = lil_matrix((nnodes,nnodes))
    for n1,d1 in G.nodes_iter(data=True):
        for n2,d2 in G.nodes_iter(data=True):
            i1 = G.nodes().index(n1)
            i2 = G.nodes().index(n2)

            if i1 == i2 and not selfloops:
                continue

            # value = criteriafn(d1,d2)
            # belief[i1,i2] = value

            if i2 > i1:
                if LINKSONLY:
                    if G.has_edge(n1,n2) or G.has_edge(n2,n1):
                        value = criteriafn(d1,d2)
                        belief[i1,i2] = value

                        if nx.is_directed(G):
                            value = criteriafn(d2,d1)
                        belief[i2,i1] = value
                else:
                    value = criteriafn(d1,d2)
                    belief[i1,i2] = value

                    if nx.is_directed(G):
                        value = criteriafn(d2,d1)
                    belief[i2,i1] = value


    print('belief: {}'.format(belief.sum()))
    return belief

def hyp_noise(matrix, noise):
    e = np.random.randint(noise*-1, noise+1,matrix.shape)
    tmp = e + matrix.copy()
    tmp[np.where(tmp < 0)] = 0.
    return csr_matrix(tmp)

######################################################################
# janus
######################################################################

def run_janus(data,isdirected,isweighted,ismultigraph,dependency,algorithm,path,kmax,klogscale,krank,**hypotheses):

    graph = DataMatrix(isdirected,isweighted,ismultigraph,dependency,algorithm,path)
    graph.dataoriginal = data
    graph.nnodes = data.shape[0]
    graph.nedges = data.sum() / (1 if isdirected else 2)
    janus = JANUS(graph, path)

    janus.createHypothesis('data')
    janus.createHypothesis('uniform')
    # janus.createHypothesis('selfloop')

    for k,v in hypotheses.items():
        janus.createHypothesis(k,v)

    janus.generateEvidences(kmax,klogscale)
    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.plotEvidences(krank,bboxx=0.8,bboxy=0.8)
    janus.plotBayesFactors(krank,bboxx=0.75,bboxy=0.7)
    janus.saveReadme()

def run_hyptrails(kmax,data,path,name,**hypotheses):
    ht = HypTrails()
    #ht.fit(sequences)

    ht.state_count = data.shape[0]
    ht.transitions = data
    evidences = {}

    hyp_data = ht.transitions.copy()
    hyp_uniform = csr_matrix(data.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    evidences['data'] = hyptrails_evidences(kmax, ht, hyp_data, ax,'data','o','--')
    evidences['uniform'] = hyptrails_evidences(kmax, ht, hyp_uniform, ax,'uniform','*','-')

    for k,m in hypotheses.items():
        evidences[k] = hyptrails_evidences(kmax, ht, m, ax,k,'x','-')

    rank = {k:e[len(e)-1] for k,e in evidences.items()}
    rank_sorted = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
    print('========= RANKING ==========')
    for ke in rank_sorted:
        print('- {} ({})'.format(ke[0],ke[1]))

    # Further plotting
    ax.set_xlabel("hypothesis weighting factor k")
    ax.set_ylabel("marginal likelihood / evidence (log)")

    handles, labels = ax.get_legend_handles_labels()
    t = [(l,h,rank[l]) for l,h in zip(labels, handles)]
    labels, handles, evidences = zip(*sorted(t,key=lambda t: t[2],reverse=True))
    legend = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.) # outside of the figure
    # legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.27,0.5)) # inside
    # plt.legend(bbox_to_anchor=(1,0.8),loc="upper right", handlelength = 3)

    plt.grid(False)
    ax.xaxis.grid(True)

    fn = os.path.join(path,name)
    plt.savefig(fn)
    plt.close()

def hyptrails_evidences(kmax, ht, hypothesis, ax,name,marker,linestyle):
    evidences = []
    # r = np.sort(np.unique(np.append(np.logspace(-1,kmax,kmax+2),1)))
    r = range(0,kmax+1)
    for k in r:
        if k == 0:
            evidences.append(ht.evidence(csr_matrix((ht.state_count,ht.state_count)),k))
        else:
            evidences.append(ht.evidence(hypothesis,k))
    ax.plot(range(len(evidences)), evidences, marker=marker, clip_on = False, label=name, linestyle=linestyle)
    return evidences

######################################################################
# main
######################################################################
if __name__ == '__main__':
    isdirected = False
    isweighted = False
    ismultigraph = True
    dependency = c.LOCAL
    algorithm = 'sociopatterns'
    kmax = 10
    klogscale=False
    krank = 10
    path = '../resources/sociopatterns-{}-{}-{}-{}/'.format(dependency,HIGH,LOW,'linksonly' if LINKSONLY else 'all')

    if not os.path.exists(path):
        os.makedirs(path)

    ### loading raw data
    df = loadCSV(path)

    ### creating networkx graph and plots
    G = create_graph(df)
    m = create_matrix(G)
    save_matrix(m,path,'data.mtx')
    plot_graph(G,path,'graph.pdf')
    plot_matrix(m,path,'matrix.pdf')

    ### creating garph-tool graph and block plots
    g = create_graphtool(G,m)
    plot_graphtool(g,g.vp.sex,path,'graph_gender.pdf')
    plot_graphtool(g,g.vp.age,path,'graph_age.pdf')
    plot_graphtool(g,g.vp.household,path,'graph_household.pdf')

    plot_block_matrix(m,g,g.vp.sex,path,'block_gender.pdf')
    plot_block_matrix(m,g,g.vp.age,path,'block_age.pdf')
    plot_block_matrix(m,g,g.vp.household,path,'block_household.pdf')

    plot_blockmembership_dist(g,g.vp.sex,path,'hist_gender.pdf')
    plot_blockmembership_dist(g,g.vp.age,path,'hist_age.pdf')
    plot_blockmembership_dist(g,g.vp.household,path,'hist_household.pdf')

    h1 = build_hypothesis(G,similar_age)
    plot_matrix(h1,path,'h1_similar_age.pdf')
    save_matrix(h1,path,'h1_similar_age.mtx')

    h2 = build_hypothesis(G,same_household)
    plot_matrix(h2,path,'h2_same_household.pdf')
    save_matrix(h2,path,'h2_same_household.mtx')

    h3 = build_hypothesis(G,same_sex)
    plot_matrix(h3,path,'h3_same_gender.pdf')
    save_matrix(h3,path,'h3_same_gender.mtx')

    h4 = build_hypothesis(G,different_sex)
    plot_matrix(h4,path,'h4_different_gender.pdf')
    save_matrix(h4,path,'h4_different_gender.mtx')

    run_janus(m,isdirected,isweighted,ismultigraph,dependency,algorithm,path,kmax,klogscale,krank,
              similar_age=h1,same_household=h2,same_gender=h3,
              different_gender=h4)

