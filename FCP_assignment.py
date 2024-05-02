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
        Each node is connected to its neighbours depending on the neighbour range r.
        '''
        
        #Your code for task 4 goes here
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
        Then each edge is rewired depending on probibility p. 
        '''

        #Your code for task 4 goes here
        self.nodes = []
        neighbour_range = 2
        
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))
        
        connection = []

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if (index + neighbour_range) > N:
                    if (index + neighbour_range - N) <= neighbour_index and (index - neighbour_range) <= neighbour_index:
                        connection.append([index, neighbour_index])
                        node.connections[neighbour_index] = 1
                        self.nodes[neighbour_index].connections[index] = 1
                elif (index - neighbour_range) < N:
                     if (index + neighbour_range) <= neighbour_index and (index - neighbour_range + N) <= neighbour_index:
                        connection.append([index, neighbour_index])
                        node.connections[neighbour_index] = 1
                        self.nodes[neighbour_index].connections[index] = 1
                
                if (index + neighbour_range) >= neighbour_index and (index - neighbour_range) <= neighbour_index:
                    connection.append([index, neighbour_index])
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if search(connection, index, neighbour_index):
                    if np.random.random() < re_wire_prob:
                        flag = True
                        while flag:
                            ran = np.random.randint(0, 10)
                            flag = search(connection, index, ran)
                        connection[index] = [index, ran]
                        node.connections[neighbour_index] = 0
                        self.nodes[neighbour_index].connections[index] = 0
                        node.connections[ran] = 1
                        self.nodes[ran].connections[index] = 1

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
        
        plt.show()

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
    '''
    This function calculates the change in agreement that would result if the cell at (row, col) were to flip its value.
    Inputs:
        population (numpy array): Grid representing the population's opinions.
        row (int): Row index of the cell.
        col (int): Column index of the cell.
        external (float): External influence on the cell's opinion.
    Returns:
        change_in_agreement (float): The change in agreement.
    '''

    n_rows, n_cols = population.shape
    neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
    agreement = 0

    # Circular boundary conditions
    if row == 0:
        neighbors.append((n_rows-1, col))
    elif row == n_rows - 1:
        neighbors.append((0, col))
    if col == 0:
        neighbors.append((row, n_cols-1))
    elif col == n_cols - 1:
        neighbors.append((row, 0))
    
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
def update_opinion(input, beta, threshold, use_network=False):
    '''
    
    '''  
     
    # randomly choose one person
    if use_network:
        tmp_nodes = input.nodes.copy()
        i = np.random.randint(0, len(input.nodes))
        neighbors = np.array(input.nodes[i].connections)

        if np.count_nonzero(neighbors) > 0:
            j = np.random.choice(np.where(neighbors==1)[0])

            if abs(input.nodes[j].value - input.nodes[i].value) < threshold:
                tmp_nodes[i].value = input.nodes[i].value + beta * (input.nodes[j].value - input.nodes[i].value)
                tmp_nodes[j].value = input.nodes[j].value + beta * (input.nodes[i].value - input.nodes[j].value)
            input.nodes = tmp_nodes.copy()

        return input

    else:
        opinions = input.copy()
        tmp_opnions = input.copy()
        i = np.random.randint(0, len(opinions), 1)

        # radomly choose one neighbor from that person
        if np.random.randint(0,2,1) == 0:
            j = i - 1 if i > 0 else -1
        else:
            j = i + 1 if i < len(opinions) - 1 else 0

        # update opinion
        if abs(opinions[j] - opinions[i]) < threshold:
            tmp_opnions[i] = opinions[i] + beta * (opinions[j] - opinions[i])
            tmp_opnions[j] = opinions[j] + beta * (opinions[i] - opinions[j])
        return tmp_opnions



def defuant_main(beta=0.2,threshold=0.2):
    '''
    
    '''

    #Your code for task 2 goes here
    population = 100
    steps = 10000
    interval = 100

    opinions = np.random.rand(population)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))  # 1 row, 2 columns
    fig.suptitle(f'Coupling: {beta:.6f}, Threshold: {threshold:.6f}')
    for t in range(steps):
        opinions = update_opinion(opinions, beta, threshold)
        if t % interval == 0:
            ax1.cla()
            ax1.hist(opinions)
            ax1.set_xlim(0, 1)
            ax1.set_xlabel('opinion')

            ax2.scatter([t // interval] * population, opinions, color='red')
            ax2.set_ylim(0, 1)
            ax2.set_xlim(0, t // interval)
            ax2.set_ylabel('opinion')

            plt.pause(0.05)

    plt.show()


def test_defuant():
    '''
    
    '''

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
        axs[0, 1].set_ylabel('Opinion')
       
        axs[0, 2].cla()
        axs[0, 2].set_xlim(0, 1)
        axs[0, 2].hist(opinions2)
        
        axs[0, 3].scatter([t] * steps, opinions2, color='red')
        axs[0, 3].set_ylim(0, 1)
        axs[0, 3].set_xlim(0, t + 1)
        axs[0, 3].set_ylabel('Opinion')
       
        axs[1, 0].cla()
        axs[1, 0].set_xlim(0, 1)
        axs[1, 0].hist(opinions3)
       
        axs[1, 1].scatter([t] * steps, opinions3, color='red')
        axs[1, 1].set_ylim(0, 1)
        axs[1, 1].set_xlim(0, t + 1)
        axs[1, 1].set_ylabel('Opinion')
        
        axs[1, 2].cla()
        axs[1, 2].set_xlim(0, 1)
        axs[1, 2].hist(opinions4)
        
        axs[1, 3].scatter([t] * steps, opinions4, color='red')
        axs[1, 3].set_ylim(0, 1)
        axs[1, 3].set_xlim(0, t + 1)
        axs[1, 3].set_ylabel('Opinion')
        
        plt.pause(0.05)
    plt.show()

'''
==============================================================================================================
This section contains code for the Small World Network - task 4 in the assignment
==============================================================================================================
'''

def search(list, zero, one):
    '''
    This function searches for the connections [zero, one] in the list
    Returns true if it is and false if it is not
    '''

    for l in list:
        if l[0] == zero and l[1] == one:
            return True
        elif l[0] == one and l[1] == zero:
            return True
    return False


'''
==============================================================================================================
This section contains code for the task 5
==============================================================================================================
'''

def use_network(beta=0.5, threshold=0.5, N=100):
    '''
    
    '''
    
    population = N
    steps = 10000

    interval = 100
    network = Network()
    network.make_small_world_network(population, re_wire_prob=0.2)

    fig = plt.figure()
    for t in range(steps):
        network = update_opinion(network, beta, threshold, use_network=True)
        if t % interval == 0:
            # plt.cla()
            ax = fig.add_subplot(111)
            ax.set_axis_off()

            num_nodes = len(network.nodes)
            network_radius = num_nodes * 10
            ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
            ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

            for (i, node) in enumerate(network.nodes):
                node_angle = i * 2 * np.pi / num_nodes
                node_x = network_radius * np.cos(node_angle)
                node_y = network_radius * np.sin(node_angle)

                circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
                ax.add_patch(circle)

                for neighbour_index in range(i + 1, num_nodes):
                    if node.connections[neighbour_index]:
                        neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                        neighbour_x = network_radius * np.cos(neighbour_angle)
                        neighbour_y = network_radius * np.sin(neighbour_angle)

                        ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

            plt.pause(0.3)

    plt.show()


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def parse_command_line_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-ising_model', action='store_true', help='Enable Ising model simulation')
    parser.add_argument('-external', type=float, default=0.0, help='External pull value')
    parser.add_argument('-alpha', type=float, default=1.0, help='Tempurature parameter')
    parser.add_argument('-test_ising', action='store_true', help='Run Ising model test')
    parser.add_argument('-defuant', action='store_true', help='Enable Deffuant model simulation')
    parser.add_argument('-beta', type=float, default=0.2, help='Beta parameter for the Deffuant model')
    parser.add_argument('-threshold', type=float, default=0.2, help='Threshold parameter for the Deffuant model')
    parser.add_argument('-test_defuant', action='store_true', help='Run Deffuant model test')
    parser.add_argument('-network', type=int, help='General network simulation')
    parser.add_argument('-test_network', action='store_true', help='Run general network test')
    parser.add_argument('-ring_network', type=int, help='Create a ring network of specified size')
    parser.add_argument('-small_world', type=int, help='Create a small-world network of specified size')
    parser.add_argument('-re_wire', type=float, default=0.2, help='Rewiring probability for the small-world network')
    parser.add_argument('-use_network', type=int, help='')

    return parser.parse_args()

def main():
    args = parse_command_line_arguments()

    #task 1
    if args.ising_model:
        population = np.random.choice([-1, 1], size=(100, 100))
        ising_main(population, args.alpha, args.external)
    elif args.test_ising:
        test_ising()
    
    #task2
    if args.defuant:
        if args.use_network: #task 5
            use_network(N=args.use_network)
        else:
            defuant_main(args.beta, args.threshold)
    elif args.test_defuant:
        test_defuant()

    #task 4
    if args.ring_network:
        ring = Network()
        ring.make_ring_network(args.ring_network)
        ring.plot()
    elif args.small_world:
        small = Network()
        small.make_small_world_network(args.small_world, args.re_wire)
        small.plot()

    if not any([args.defuant, args.test_defuant, args.ising_model, args.test_ising, args.ring_network, args.small_world]):
        print('Please use -h or --help for command usage details.')

if __name__ == "__main__":
    main()
