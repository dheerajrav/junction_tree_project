import networkx as nx
import random
import itertools
import ProbPy
import pandas as pd


def only_prob(df_colm):
    prob = []
    val_count = df_colm.value_counts()
    indices = val_count.index.to_list()
    indices.sort()
    for ind in indices:
        prob.append(float(val_count[ind]/len(df_colm)))

    return prob


def conditional_prob_list(df, var_list1, var_list2):
    # var_list2 given var_list1 list
    num = df.groupby(var_list1+var_list2[:-1])[var_list2[-1]].value_counts()
    dem = 0
    denom = df.groupby(var_list1)[var_list1[0]].count()
    cond_prob_val = num / denom
    cond_prob_val.sort_index(inplace=True)
    cond_prob_val_fl = [float(val) for val in cond_prob_val.values]
    return cond_prob_val_fl


class JunctionTree:
    """
    Takes a clique graph, original graph
    as input and generates the appropriate
    junction tree.

    Must also take the potential data as
    input and generate

    """

    def __init__(self, c_graph, o_graph, df):

        # Networkx graph
        self.clique_graph = c_graph
        self.o_graph = o_graph
        self.j_graph = None
        self.data = df

        self.Z = 0
        self.rand_vars = {}
        self.generate_potentials()


    def fetch_factor(self, node, edge_nodes):
        """
            Not only do you have to generate the proper
            potential, but you have to generate the proper
            values for the factor as well
        """

        factor_vars = [node]
        if len(edge_nodes) > 0:
            factor_vars = [node] + edge_nodes


        factor_list = [self.rand_vars[rand_var] for rand_var in factor_vars]
        factor_domains = [self.rand_vars[var].domain for var in factor_vars]

        if len(edge_nodes) == 0:
            cond_prob = only_prob(self.data[node])
        else:
            cond_prob = conditional_prob_list(self.data, [node], edge_nodes)

        factor = ProbPy.Factor(factor_list, cond_prob)

        return factor


    def generate_potentials(self):
        """
        A good belief network must not contain cycles.
        Therefore, if there are cycles, the code must fail

        Now, iterate through the graph nodes and check its
        incoming edges and generate the potentials.
        """

        try:
            nx.algorithms.cycles.find_cycle(G)
            nx.exception.HasACycle("This belief network has a cycle. Check yo-self")
        except nx.exception.NetworkXNoCycle:
            pass

        potentials = []
        for node in self.o_graph.nodes():

            node_domain = list(self.data[node].unique())  # Random variables domain
            node_domain = [int(_) for _ in node_domain]  # Hacky fix
            node_domain.sort()

            rand_var = ProbPy.RandVar(node, node_domain)
            self.rand_vars[node] = rand_var

        for node in self.o_graph.nodes():
            in_edges = [in_edge_node for in_edge_node, _ in self.o_graph.in_edges(node)]

            factors = self.fetch_factor(node, in_edges)

            # Create ProbPy potentials
            potentials.append(factors)

        self.o_graph.graph['potentials'] = potentials

    def generate_junction_tree(self):
        """
        When called, generates the junction tree
        by

        1. Getting the maximal clique - DONE
        2. Generating the junction_graph by iterating over cliques - DONE
        3. Get the Maximal Spanning Tree of the Junction Graph to form the JT - DONE
        """

        cliques = list(nx.find_cliques(self.clique_graph))
        self.j_graph = nx.Graph()
        for index, clique in enumerate(cliques):
            self.j_graph.add_node('c_{}'.format(index+1))
            self.j_graph.nodes['c_{}'.format(index+1)]['nodes'] = clique

        edge_iter = 1
        for clique_1 in self.j_graph.nodes():
            for clique_2 in self.j_graph.nodes():
                c1 = self.j_graph.nodes[clique_1]['nodes']
                c2 = self.j_graph.nodes[clique_2]['nodes']
                separator = list(set(c1).intersection(set(c2)))
                if len(separator) > 0:
                    edge_name = "s_{}".format(edge_iter)
                    edge_iter += 1

                    self.j_graph.add_edge(clique_1, clique_2, name=edge_name, weight=len(separator), nodes=separator)

        # Now get the maximal spanning tree
        # print edges
        self.j_tree = nx.algorithms.tree.maximum_spanning_tree(self.j_graph)

    def assign_potentials(self):
        """
        As mentioned in the book, iterate over nodes of the junction tree

        1. Assign edge potential to be 1
        2. Assign node potentials to be such that the original potential variables
        form a subset of the junction tree node variables
        """

        graph_potentials = self.o_graph.graph['potentials']
        randomized_potentials = random.sample(graph_potentials, len(graph_potentials))

        for node in self.j_tree.nodes():
            self.j_tree.nodes[node]['potential'] = []

        for potential in randomized_potentials:
            potential_set = set([rv.name for rv in potential.rand_vars])
            for node in self.j_tree.nodes():
                clique_node_set = self.j_tree.nodes[node]['nodes']
                if potential_set.issubset(clique_node_set):
                    self.j_tree.nodes[node]['potential'].append(potential)
                    break

    def sum_product(self):
        """
        This is the Junction Tree Algorithm
        That calculates the potential for various values
        of the original nodes

        Shafer-Shenoy Propagation

        1. Assign Potentials
        """

        # Assign potentials

        self.assign_potentials()

        # Pick a random node as root
        # Evidence
        root_node = random.sample(list(self.j_tree.nodes()), 1)[0]

        # Now collect potentials (post-order)
        for neighbor in self.j_tree.neighbors(root_node):
            self.collect(root_node, neighbor)

        # Now distribute potentials (pre-order)
        for neighbor in self.j_tree.neighbors(root_node):
            self.distribute(root_node, neighbor)

        # Now compute marginals
        for clique in self.j_tree.nodes:
            clique_marginal = self.compute_marginal(clique)
            self.j_tree.nodes[clique]['marginal'] = clique_marginal

            # Compute the marginalizing constant
            self.Z += sum(clique_marginal.values)

        # Now iterate over the marginals and noramlise them

        for node in self.j_tree.nodes:
            self.j_tree.nodes[node]['probability'] = self.j_tree.nodes[node]['marginal'] / self.Z
            print(self.j_tree.nodes[node]['probability'])


    def collect(self, node_i, node_j):
        """
        Post Order Traversal
        """
        for neighbor in self.j_tree.neighbors(node_j):
            if neighbor != node_i:
                self.collect(node_j, neighbor)

        self.send_message(node_j, node_i)

    def distribute(self, node_i, node_j):
        """
        Pre Order Traversal
        """

        self.send_message(node_i, node_j)
        for neighbor in self.j_tree.neighbors(node_j):
            if neighbor != node_i:
                self.distribute(node_j, neighbor)

    def send_message(self, node_j, node_i):
        """
        This part does the message passing.

        Firstly, it computes the potential of the node by taking
        the product of the individual potentials stored in the
        node attribute of j

        Secondly, it loops over all the neighbors of node_j sans node_i
        and computes the incoming potential from them to node_j

        Thirdly, it marginalizes over the nodes in j not in i
        and stores that as a message
        """

        nodes_cj = set(self.j_tree.nodes[node_j]['nodes'])
        nodes_ci = set(self.j_tree.nodes[node_i]['nodes'])
        sum_rand_vars = nodes_cj.difference(nodes_ci)
        product_potential = None

        # This generates clique potential from node potential
        for potential in self.j_tree.nodes[node_j]['potential']:
            if product_potential is None:
                product_potential = potential
            else:
                product_potential = product_potential * potential

        # Now that product potential is defined, get the message passed potentials
        # Fetch the neighbors of node_i not

        other_neighbors_j = list(self.j_tree.neighbors(node_j))
        other_neighbors_j.remove(node_i)

        for other_neighbor_j in other_neighbors_j:
            product_potential = product_potential * self.j_tree.edges[(other_neighbor_j, node_j)]['{}_{}'.format(other_neighbor_j, node_j)]

        # Now marginalize
        potential_rand_vars = set([rv.name for rv in product_potential.rand_vars])

        marginal_rand_vars = [self.rand_vars[mrv] for mrv in potential_rand_vars.difference(sum_rand_vars)]

        lambda_j_i = product_potential.marginal(marginal_rand_vars)

        self.j_tree.edges[(node_j, node_i)]['{}_{}'.format(node_j, node_i)] = lambda_j_i

    def compute_marginal(self, node):
        """
        This is where belief network potential
        for the junction tree nodes are calculated
        """

        marginal_potential = None

        # Clique potential from node potential
        for potential in self.j_tree.nodes[node]['potential']:
            if marginal_potential is None:
                marginal_potential = potential
            else:
                marginal_potential = marginal_potential * potential

        for neighbor in self.j_tree.neighbors(node):
            marginal_potential = marginal_potential * self.j_tree.edges[(neighbor, node)]['{}_{}'.format(neighbor, node)]

        return(marginal_potential)


if __name__ == '__main__':

    G = nx.read_edgelist("og_edges.txt", delimiter=",", create_using=nx.DiGraph)
    CG = nx.read_edgelist("moralised_edges.txt", delimiter=",")
    df = pd.read_csv("processed_plus_one.csv")

    JTA = JunctionTree(CG, G, df)
    JTA.generate_junction_tree()

    # Do Shafer-Shenoy Propagation
    JTA.sum_product()
