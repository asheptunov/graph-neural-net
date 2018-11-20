import java.util.*;

public class NeuralNet {
	// new structure
	private int depth; // # of neuron layers
	private int inputDim, outputDim; // # of neurons in input and output layer, respectively
	private int hiddenLayerDim; // # of neurons in arbitrary hidden layer
	private List<double[]> activations;
	private List<double[][]> weights;

	// functions for math
	private Random random;
	private Sigmoid sigmoidFunc;
	private SigmoidPrime sigmoidPrime;
	private LossFunction lossFunc;
	private LossFunctionPrime lossPrime;

	// Abstraction function:
	// A NeuralNet represents the ADT of a neural network with depth # of layers and a depth - 1 # of weight layers.
	// These correspond to the lengths of arrays activations and weights, respectively. The net has an input
	// dimension of inputDim, and an output dimension of outputDim. These correspond to the lengths of the first
	// and last elements in array activations, respectively.
	// Array activations[i] refers to the vector of neuron activations in layer i.
	// Value activations[i][j] refers to the value of the jth neuron in layer i.
	// Matrix weights[i] refers to the matrix of all edge weights between the ith and i+1th activation layers.
	// Array weights[i][j] refers to the vector of edge weights originating from the jth neuron in the ith activation layer.
	// Value weights[i][j][k] refers to the value of the edge weight from the jth neuron in the ith activation layer
	// to the kth neuron in the i+1th activation layer.
	//
	// Examples:
	// A depth 1 neural net will have matching inputDim and outputDim, have a single element in activations, and an
	// empty weights array. It will have no hidden layers.
	// A depth 2 neural net will have 2 elements in activations, and 1 element in weights. It will have no hidden layers.
	// A depth 3 neural net will have 3 elements in activations, and 2 elements in weights. It will have 1 hidden layer
	// of dimension hiddenLayerDim.

	// Representation invariant:
	// depth, inputDim, outputDim, hiddenLayerDim > 0
	// activations, weights, random, sigmoidFunc, sigmoidPrime, lossFunc, lossPrime != null
	// activations[...], weights[...], weights[...][...] != null
	// activations.length == depth
	// weights.length == depth - 1
	// activations[0].length == inputDim
	// activations[depth - 1].length == outputDim
	// weights[i].length == activations[i].length
	// weights[i][j].length == activations[i + 1].length
	// depth > 2 -> activations[1...(depth - 2)].length == hiddenLayerDim

	// Graphic:
	//
	//     a_1          w_1,1       b_1                 z_1
	//                  w_1,2
	//                  ...
	//                  w_1,hld
	//     a_2          w_2,1       b_2                 z_2
	//                  w_2,2
	//                  ...
	//                  w_2,hld                 .
	//     ...          w_?,1       ...         .       ...
	//                  w_?,2                   .
	//                  ...
	//                  w_?,hld
	//  a_inputDim      w_id,1      b_hld           z_outputDim
	//                  w_id,2
	//                  ...
	//                  w_id,hld
	//
	//      ^              ^         ^                   ^
	// activations[0] weights[0] activations[1]   activations[depth-1]
	//

	/**
	 * Constructs a new neural net with some input, output, and hidden layer dimensions, as well as some depth (layer
	 * count). A depth 1 neural net will perform no modifications on input and output data, and is assumed to have equal
	 * input and output dimensions. All edges will have random edge weights ranging from 0.0-1.0, and all neurons will
	 * have data set to 0.0. If hidden layers exist, all will have the same specified dimension. All layers that aren't
	 * the input or output layer will be hidden layers. Sets the function to use during forward propagation to the
	 * specified sigmoid, and the sets its derivative to sigmoid prime for use during back propagation of error. Does the
	 * same for a given loss function and its derivative, which will be used during error calculation and back propagation,
	 * respectively.
	 * Depth, input dimension, and output dimension are assumed to be greater than zero.
	 * Sigmoid and sigmoid derivative functions are assumed to be non-null, sigmoid is assumed to represent a continuous
	 * and differentiable sigmoid function, and sigmoid prime is assumed to correctly express the derivative of the given
	 * sigmoid function. The same criteria are assumed for the given loss function.
	 *
	 * @param inputDim the dimension of input data
	 * @param outputDim the dimension of output data
	 * @param hiddenLayerDim the dimension of an arbitrary hidden layer
	 * @param depth the depth of the net
	 * @param sig the sigmoid function to apply during propagation
	 * @param sigPrime the derivative of the sigmoid function to apply during back propagation
	 * @param loss the loss function to apply during error evaluation
	 * @param lossPrime the derivative of the loss function to apply during back propagation
	 */
	public NeuralNet(int inputDim, int outputDim, int hiddenLayerDim, int depth,
	                 Sigmoid sig, SigmoidPrime sigPrime, LossFunction loss, LossFunctionPrime lossPrime) {
		assert depth > 0 && inputDim > 0 && outputDim > 0 && hiddenLayerDim > 0; // ge checks
		assert (depth != 1) || (inputDim == outputDim); // equiv to (depth == 1) -> (inputDim == outputDim)
		assert sig != null && sigPrime != null && loss != null && lossPrime != null; // non-null checks

		// structure init
		this.inputDim = inputDim;
		this.outputDim = outputDim;
		this.hiddenLayerDim = hiddenLayerDim;
		this.depth = depth;
		activations = new ArrayList<>();
		weights = new ArrayList<>();

		// build
		appendLayer(inputDim);
		int i; // need i outside of for scope
		for (i = depth; i > 1; i--) {
			appendLayer(hiddenLayerDim);
		}
		if (i == 1) appendLayer(outputDim);

		// function init
		random = new Random(1);
		this.sigmoidFunc = sig;
		this.sigmoidPrime = sigPrime;
		this.lossFunc = loss;
		this.lossPrime = lossPrime;

		checkRep();
	}

	/**
	 * Appends a new layer of the specified number of neurons to the neural net, and links all new nodes with all the
	 * nodes from the previous layer (if such exists) using a cartesian product of edges. New edges will have random edge
	 * weights ranging from 0.0-1.1, and new neurons will have data set to 0.0.
	 * Assumes the input dimension is greater than zero.
	 *
	 * @param dim the dimension of the layer to construct
	 */
	private void appendLayer(int dim) {
		assert dim > 0;

		double[] newNeurons = new double[dim]; // all values automatically 0.0
		if (!activations.isEmpty()) { // we are adding a hidden layer or output layer; add weights
			double[][] newWeights = new double[activations.get(activations.size() - 1).length][dim];
			for (int i = 0; i < newWeights.length; i++) {
				for (int j = 0; j < dim; j++) {
					newWeights[i][j] = random.nextDouble();
				}
			}
			weights.add(newWeights);
		}
		activations.add(newNeurons);
		checkRep();
	}

	/**
	 * Propagates the input vector down the layers of the neural net, setting neuron values to those calculated from edge
	 * weights and smoothing functions. Input vector dimension is assumed to match the input dimension of the neural net.
	 * Returns this net.
	 *
	 * @param input the input vector to propagate
	 * @return this net
	 */
	public NeuralNet propagate(double[] input) {
		assert input.length == inputDim;
		System.arraycopy(input, 0, layer0, 0, layer0.length); // assign input layer
		for (int l = 1; l < depth; i++) { // traverse through output layer multiplying layer vector by weights matrix
			double[] pActivations = activations.get(l - 1); // l-1th layer activations (previous)
			double[] lActivations = activations.get(l); // lth layer activations (current)
			double[][] lWeights = weights.get(l - 1); // weights between l-1th layer (previous) and lth layer (current)
			assert lWeights.length == pActivations.length;
			assert lWeights[0].length == lActivations.length;
			for (int i = 0; i < lActivations.length; i++) {
				lActivations[i] = 0.0; // reset all values
			}
			for (int i = 0; i < pActivations.length; i++) {
				for (int j = 0; j < lActivations.length; j++) {
					// propagates one value from previous layer into all values from next layer
					lActivations[j] += (pActivations[i] * lWeights[i][j]);
				}
			}
			for (int i = 0; i < lActivations.length; i++) {
				lActivations[i] = sigmoidFunc.func(lActivations[i]); // apply sigmoid to all values
			}
		}
		checkRep();
		return this;
	}

	/**
	 * Returns the number of layers in the neural net, composed of 1 input layer, 0-1 output layers, and any non-negative
	 * number of hidden layers.
	 *
	 * @return
	 */
	public int depth() {
		return depth;
	}

	public int getOutput() {
		return 0; // TODO
	}

	/**
	 * Calculates and returns a loss for a given input set by forward propagating the input, comparing it to the
	 * provided expected output and applying the given loss function per output component, and then aggregating the
	 * components.
	 * Assumes input list is non-null and has same dimension as the net's input dimension. Assumes the actual list is
	 * non-null and has the same dimension as the net's output dimension. Assumes the loss function is non-null.
	 *
	 * @param input    a single element of input / training data
	 * @param expected the expected output to calculate loss against
	 * @param loss     the loss function to compute loss with
	 * @return the loss vector for the output
	 */
	public double calculateLoss(List<Double> input, List<Double> expected, LossFunction loss) {
		assert input.size() == inputDim;
		assert expected.size() == outputDim;
		assert loss != null;
		List<Double> calculated = propagate(input);
		double output = 0;
		for (int i = 0; i < calculated.size(); i++) {
			output += loss.loss(calculated.get(i), expected.get(i));
		}
		checkRep();
		return output;
	}

	/**
	 * Calculates and returns a gradient representing dL/dw, or the gradient of the loss (error) with respect to all the
	 * weights in the net. This works by first propagating the input through the net, calculating the loss from it using
	 * the given loss function, and then applying a propagation algorithm using the neural net's sigmoid prime function
	 * and the given loss prime function. The list at output.size() will be one less than the number of layers in this
	 * net; output.get(n) will represent the dL/dw for each weight between layer n and layer n+1; output.get(n)[i] will
	 * represent all the weights starting at neuron i in layer n; output.get(n)[i][j] will represent the weight from the
	 * ith neuron in layer n to the jth weight in layer n+1.
	 * Loss function and loss function derivative are assumed to be non-null, and loss function is assumed to be
	 * continuous and differentiable. Input and expected lists are expected be non-null; input list should match the input
	 * dimension for the neural net; expected list should match the output dimension of the net.
	 *
	 * @param input         a single element of input / training data
	 * @param expected      the expected output to calculate loss against
	 * @param lossFunction  the loss function to compute loss with
	 * @param lossFuncPrime the derivative of the loss function
	 * @return the list of error gradients with respect to weights
	 */
	public List<double[][]> calculateWeightGradient(List<Double> input, List<Double> expected,
	                                                  LossFunction lossFunction, LossFunctionPrime lossFuncPrime) {
		List<double[][]> weightLayers = new ArrayList<>(layers.size() - 1);
		double loss = calculateLoss(input, expected, lossFunction); // propagates input, calculates loss
		for (int l = 1; l < layers.size(); l++) { // start at 1 because layer 0 is inputs, they have no parents
			List<Node<Double, Double>> layer = layers.get(l);
			double[][] gradMatrix = new double[layers.get(l - 1).size()][layers.get(l).size()];
			weightLayers.add(gradMatrix);
			for (int i = 0; i < layer.size(); i++) {
				Node<Double, Double> child = layer.get(i);
				Set<Node<Double, Double>> parents = child.parents();
				for (int j = 0; i < parents.size(); j++) {
					// oh shit, parents is a set
				}
			}
		}


		checkRep();
		return null; // TODO
	}

	/**
	 * Checks the representation invariant
	 */
	private void checkRep() {
		assert !layers.isEmpty(); // no zero dimensional nets
		assert layers.get(0).size() == inputDim; // first layer matches input dimension
		assert layers.get(layers.size() - 1).size() == outputDim; // last layer matches output dimension
	}

}
