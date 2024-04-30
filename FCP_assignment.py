import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

class Node:

    def __init__(self, value, number, connections=None):

        self.index = number
        self.connections = connections
        self.value = value

class Network: 

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes 

    def get_mean_degree(self):
        #Your code  for task 3 goes here
        pass
    def get_mean_clustering(self):
        #Your code for task 3 goes here
        pass
    def get_mean_path_length(self):
        #Your code for task 3 goes here
        pass
    def make_random_network(self, N, connection_probability=0.5):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=1):
        '''
        This function makes a *ring* network of size N.
        Each node is connected to its neighbours depending on the neighbour range n.
        '''
        
        #Your code  for task 4 goes here
        
        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if (index + neighbour_range) > N:
                    if (index + neighbour_range - N) <= neighbour_index and (index - neighbour_range) <= neighbour_index:
                        node.connections[neighbour_index] = 1
                        self.nodes[neighbour_index].connections[index] = 1
                elif (index - neighbour_range) < N:
                     if (index + neighbour_range) <= neighbour_index and (index - neighbour_range + N) <= neighbour_index:
                        node.connections[neighbour_index] = 1
                        self.nodes[neighbour_index].connections[index] = 1

                if (index + neighbour_range) >= neighbour_index and (index - neighbour_range) <= neighbour_index:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_small_world_network(self, N, re_wire_prob=0.2):
        '''
        This function makes a *small world* network of size N. 
        Each node if first connected to its 2 closest neighbours. 
        Then each node is rewired depending on probibility p. 
        '''
        self.nodes = []

        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if np.random.random() < re_wire_prob:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

        neighbour_range = 2

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if (index + neighbour_range) > N:
                    if (index + neighbour_range - N) <= neighbour_index and (index - neighbour_range) <= neighbour_index:
                        if node.connections[neighbour_index] == 1 and self.nodes[neighbour_index].connections[index] == 1:
                            node.connections[neighbour_index] = 0
                            self.nodes[neighbour_index].connections[index] = 0
                        else:
                            node.connections[neighbour_index] = 1
                            self.nodes[neighbour_index].connections[index] = 1

                elif (index - neighbour_range) < N:
                     if (index + neighbour_range) <= neighbour_index and (index - neighbour_range + N) <= neighbour_index:
                        if node.connections[neighbour_index] == 1 and self.nodes[neighbour_index].connections[index] == 1:
                            node.connections[neighbour_index] = 0
                            self.nodes[neighbour_index].connections[index] = 0
                        else:
                            node.connections[neighbour_index] = 1
                            self.nodes[neighbour_index].connections[index] = 1


                if (index + neighbour_range) >= neighbour_index and (index - neighbour_range) <= neighbour_index:
                    if node.connections[neighbour_index] == 1 and self.nodes[neighbour_index].connections[index] == 1:
                        node.connections[neighbour_index] = 0
                        self.nodes[neighbour_index].connections[index] = 0
                    else:
                        node.connections[neighbour_index] = 1
                        self.nodes[neighbour_index].connections[index] = 1

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i+1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

    #Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number-1)%num_nodes] = 1
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert(network.get_mean_degree()==2), network.get_mean_degree()
    assert(network.get_clustering()==0), network.get_clustering()
    assert(network.get_path_length()==2.777777777777778), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert(network.get_mean_degree()==1), network.get_mean_degree()
    assert(network.get_clustering()==0),  network.get_clustering()
    assert(network.get_path_length()==5), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
    assert(network.get_clustering()==1),  network.get_clustering()
    assert(network.get_path_length()==1), network.get_path_length()

    print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
    """
    This function calculates the change in agreement that would result if the cell at (row, col) were to flip its value.
    Inputs:
        population (numpy array): Grid representing the population's opinions.
        row (int): Row index of the cell.
        col (int): Column index of the cell.
        external (float): External influence on the cell's opinion.
    Returns:
        change_in_agreement (float): The change in agreement.
    """
    n_rows, n_cols = population.shape
    neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
    agreement = 0

    for r, c in neighbors:
        if 0 <= r < n_rows and 0 <= c < n_cols:
            agreement += population[row, col] * population[r, c]

    return agreement + external * population[row, col]


def ising_step(population, external=0.0, alpha=1.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''
    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col  = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external)


    if np.random.rand() < np.exp(-agreement / alpha):
        population[row, col] *= -1

	    
	
def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)

def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
    assert(calculate_agreement(population,1,1, -10)==14), "Test 10"

    print("Tests passed")

def ising_main(population, alpha=None, external=0.0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external, alpha)
        print('Step:', frame, end='\r')
        plot_ising(im, population)





'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''
def update_opinion(opinions, beta, threshold):
    tmp_opinions = opinions.copy()
    for i in range(len(opinions)):
        left = i - 1 if i > 0 else -1
        right = i + 1 if i < len(opinions) - 1 else 0

        if abs(opinions[left] - opinions[i]) < threshold:
            # x_i(t+1) = x_i(t) + beta * (x_j(t) - x_i(t))
            tmp_opinions[i] = opinions[i] + beta * (opinions[left] - opinions[i])
            tmp_opinions[left] = opinions[left] + beta * (opinions[i] - opinions[left])
        if abs(opinions[right] - opinions[i]) < threshold:
            tmp_opinions[i] = opinions[i] + beta * (opinions[right] - opinions[i])
            tmp_opinions[right] = opinions[right] + beta * (opinions[i] - opinions[right])
    return tmp_opinions
def defuant_main(beta=0.2,threshold=0.2):
    #Your code for task 2 goes here

    population = 100
    steps = 100

    opinions = np.random.rand(population)

    # plt.figure(figsize=(6, 10))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))  # 1 row, 2 columns

    for t in range(steps):
        opinions = update_opinion(opinions, beta, threshold)

        ax1.cla()
        ax1.hist(opinions)
        ax1.set_xlim(0,1)
        ax1.set_xlabel('opinion')

        ax2.scatter([t] * steps, opinions, color='red', alpha=0.6)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, t + 1)
        ax2.set_ylabel('opinion')
        fig.suptitle(f'Coupling: {beta:.6f}, Threshold: {threshold:.6f}')
        plt.pause(0.2)
    plt.show()
def test_defuant():
    #Your code for task 2 goes here
    population = 100
    steps = 100

    opinions = np.random.rand(population)

    beta1 = 0.5
    beta2 = 0.1

    threshold1 = 0.5
    threshold2 = 0.1
    threshold3 = 0.2

    opinions1 = opinions
    opinions2 = opinions
    opinions3 = opinions
    opinions4 = opinions

    fig, axs = plt.subplots(2, 4, figsize=(12, 10))  # 1 row, 2 columns
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.2)
    for t in range(steps):
        opinions1 = update_opinion(opinions1, beta1, threshold1)
        opinions2 = update_opinion(opinions2, beta2, threshold1)
        opinions3 = update_opinion(opinions3, beta1, threshold2)
        opinions4 = update_opinion(opinions4, beta2, threshold3)

        axs[0, 0].cla()
        axs[0, 0].set_xlim(0, 1)
        axs[0, 0].hist(opinions1)

        axs[0, 1].scatter([t] * steps, opinions1, color='red')
        axs[0, 1].set_ylim(0, 1)
        axs[0, 1].set_xlim(0, t + 1)
        axs[0, 1].set_ylabel('opinion')

        axs[0, 2].cla()
        axs[0, 2].set_xlim(0, 1)
        axs[0, 2].hist(opinions2)

        axs[0, 3].scatter([t] * steps, opinions2, color='red')
        axs[0, 3].set_ylim(0, 1)
        axs[0, 3].set_xlim(0, t + 1)
        axs[0, 3].set_ylabel('opinion')

        axs[1, 0].cla()
        axs[1, 0].set_xlim(0, 1)
        axs[1, 0].hist(opinions3)

        axs[1, 1].scatter([t] * steps, opinions3, color='red')
        axs[1, 1].set_ylim(0, 1)
        axs[1, 1].set_xlim(0, t + 1)
        axs[1, 1].set_ylabel('opinion')

        axs[1, 2].cla()
        axs[1, 2].set_xlim(0, 1)
        axs[1, 2].hist(opinions4)

        axs[1, 3].scatter([t] * steps, opinions4, color='red')
        axs[1, 3].set_ylim(0, 1)
        axs[1, 3].set_xlim(0, t + 1)
        axs[1, 3].set_ylabel('opinion')

        plt.pause(0.05)
    plt.show()

'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def parse_command_line_argument():
    #Create Parser object
    parser = argparse.ArgumentParser()
    #Add arguments
    #Task 1
    parser.add_argument('-ising_model', action='store_true')
    parser.add_argument('-external', type=float, default=0.0)
    parser.add_argument('-alpha', type=float, default=1.0)
    parser.add_argument('-test_ising', action='store_true')
    
    #Task 2
    parser.add_argument('-defuant', action='store_true', help='using a value of beta = 0.2 and a threshold of 0.2')
    parser.add_argument('-beta', type=float, default=0.2, help='Allow a user to set the value of beta')
    parser.add_argument('-threshold', type=float, default=0.2, help='Allow a user to set the value of threshold')
    parser.add_argument('-test_defuant', action='store_true',
                        help='Allow s user to see initially random distribution of opinions begin to separate into distinct clusters')
    
    #Task 3
    parser.add_argument('-network', type=int)
    parser.add_argument('-test_network', action='store_true')
    
    #Task 4
    parser.add_argument('-ring_network', type=int) #ring network
    parser.add_argument('-small_world', type=int) #small world
    parser.add_argument('-re_wire', type=float, default=0.2) #re wire

    #Task 5
    parser.add_argument('-use_network', type=int)

    args = parser.parse_args()

    ising_model = args.ising_model
    external = args.external
    alpha = args.alpha
    test_ising = args.test_ising

    defuant = args.defuant
    beta = args.beta
    threshold = args.threshold
    test_defuant = args.test_defuant

    network = args.network
    test_network = args.test_network

    ring_network = args.ring_network
    small_world = args.small_world
    re_wire = args.re_wire

    use_network = args.use_network

    return ising_model, external, alpha, test_ising, defuant, beta, threshold, test_defuant, network, test_network, ring_network, small_world, re_wire, use_network

def main():

    (ising_model, external, alpha, test_ising, defuant, beta, threshold, defuant_test, network, test_network, ring_network, small_world, re_wire, network_use) = parse_command_line_argument()

    #task 1
    if ising_model:
        population = np.random.choice([-1, 1], size=(100, 100))
        ising_main(population, alpha, external)
    elif test_ising:
        test_ising()
    
    #task 2
    if defuant:
        if network_use:
            return # use network ()
        else:
            defuant_main(beta, threshold)
    elif defuant_test:
        # test
        test_defuant()
    
    #task 3
    

    #task 4
    if ring_network:
        ring = Network()
        ring.make_ring_network(ring_network)
        ring.plot()
        plt.show()
    elif small_world:
        small = Network()
        small.make_small_world_network(small_world, re_wire) #doesnt fully work yet
        small.plot()
        plt.show()


if __name__ == "__main__":
    main()