import java.util.*;

/**
 * Represents a neural network with sigmoid neurons, capable of propagating input, back propagating error, and performing
 * batch gradient descent to train the network.
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
class NeuralNet {
	// structure
	private int depth; // # of neuron layers
	private int inputDim, outputDim; // # of neurons in input and output layer, respectively
	private int hiddenLayerDim; // # of neurons in arbitrary hidden layer
	private Map<Integer, double[]> activations;
	private Map<Integer, double[][]> weights;

	// functions for math
	private Random random;
	private Sigmoid sigmoidFunc;
	private SigmoidPrime sigmoidPrime;
	private LossFunction lossFunc;
	private LossFunctionPrime lossPrime;

	// whether or not to run heavy rep inv checks
	private static final boolean DEBUG = false;
	// whether or not to run rep inv checks at all
	private static final boolean LIGHT = true;

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
	NeuralNet(int inputDim, int outputDim, int hiddenLayerDim, int depth,
	                 Sigmoid sig, SigmoidPrime sigPrime, LossFunction loss, LossFunctionPrime lossPrime) {
		assert depth > 0 && inputDim > 0 && outputDim > 0 && hiddenLayerDim > 0; // ge checks
		assert (depth != 1) || (inputDim == outputDim); // equiv to (depth == 1) -> (inputDim == outputDim)
		assert sig != null && sigPrime != null && loss != null && lossPrime != null; // non-null checks

		// function init
		random = new Random(1);
		this.sigmoidFunc = sig;
		this.sigmoidPrime = sigPrime;
		this.lossFunc = loss;
		this.lossPrime = lossPrime;

		// structure init
		this.inputDim = inputDim;
		this.outputDim = outputDim;
		this.hiddenLayerDim = (depth > 2) ? hiddenLayerDim : -1; // hidden dim is -1 if no hidden layers
		this.depth = depth;
		activations = new HashMap<>();
		weights = new HashMap<>();

		// build
		appendLayer(inputDim, 0); // layer # 0
		int i; // need i outside of for scope
		for (i = depth - 1; i > 1; i--) {
			appendLayer(hiddenLayerDim, depth - i);
		}
		if (i == 1) appendLayer(outputDim, depth - 1); // layer # depth - 1

		checkRep();
	}

	/**
	 * Appends a new layer of the specified number of neurons to the neural net, and links all new nodes with all the
	 * nodes from the previous layer (if such exists) using a cartesian product of edges. New edges will have random edge
	 * weights ranging from 0.0-1.1, and new neurons will have data set to 0.0.
	 * Assumes the input dimension is greater than zero.
	 *
	 * @param dim the dimension of the layer to construct
	 * @param layerIndex the index of the layer in the network
	 */
	private void appendLayer(int dim, int layerIndex) {
		assert dim > 0;
		double[] newNeurons = new double[dim]; // all values automatically 0.0
		if (layerIndex > 0) { // we are adding a hidden layer or output layer; add weights
			double[][] newWeights = new double[activations.get(layerIndex - 1).length][dim];
			for (int i = 0; i < newWeights.length; i++) {
				for (int j = 0; j < dim; j++) {
					newWeights[i][j] = random.nextDouble();
				}
			}
			weights.put(layerIndex - 1, newWeights);
		}
		activations.put(layerIndex, newNeurons);
	}

	/**
	 * Propagates the input vector down the layers of the neural net, setting neuron values to those calculated from edge
	 * weights and smoothing functions. Returns the output vector after propagation.
	 * Input vector is assumed to be non-null, and its dimension is assumed to match the net's input dimension.
	 *
	 * @param input the input vector to propagate
	 * @return the output vector
	 */
	double[] propagate(double[] input) {
		assert input != null;
		assert input.length == inputDim;
		System.arraycopy(input, 0, activations.get(0), 0, inputDim); // copy into first layer
		for (int l = 1; l < depth; l++) { // traverse through output layer multiplying layer vector by weights matrix
			double[] previousNeurons = activations.get(l - 1); // l-1th layer activations (previous)
			double[] currentNeurons = activations.get(l); // lth layer activations (current)
			double[][] intermediateWeights = weights.get(l - 1); // weights between l-1th layer (previous) and lth layer (current)
			assert intermediateWeights.length == previousNeurons.length;
			assert intermediateWeights[0].length == currentNeurons.length;
			for (int i = 0; i < currentNeurons.length; i++) {
				currentNeurons[i] = 0.0; // reset all values
			}
			for (int i = 0; i < previousNeurons.length; i++) {
				for (int j = 0; j < currentNeurons.length; j++) {
					// propagates one value from previous layer into all values from next layer
					currentNeurons[j] += (sigmoidFunc.func(previousNeurons[i]) * intermediateWeights[i][j]);
				}
			}
//			for (int i = 0; i < currentNeurons.length; i++) {
//				currentNeurons[i] = sigmoidFunc.func(currentNeurons[i]); // apply sigmoid to all values
//			}
		}
		double[] outputNeurons = activations.get(depth - 1);
		double[] outputActivations = new double[outputDim];
		for (int i = 0; i < outputDim; i++) {
			outputActivations[i] = sigmoidFunc.func(outputNeurons[i]);
		}
//		System.arraycopy(activations.get(depth - 1), 0, output, 0, outputDim); // copy out last layer
		checkRep();
		return outputActivations;
	}

	/**
	 * Calculates and returns a loss for a given input set by forward propagating the input, comparing it to the
	 * provided expected output and applying the net's loss function per output component, and then aggregating the
	 * components.
	 * Assumes input list is non-null and has same dimension as the net's input dimension. Assumes the actual list is
	 * non-null and has the same dimension as the net's output dimension.
	 *
	 * @param input    a single element of input / training data
	 * @param expected the expected output to calculate loss against
	 * @return the loss vector for the output
	 */
	double calculateLoss(double[] input, double[] expected) {
		assert input != null;
		assert expected != null;
		assert input.length == inputDim;
		assert expected.length == outputDim;
		double[] calculatedActivations = propagate(input);
		double loss = 0;
		for (int i = 0; i < outputDim; i++) {
			loss += lossFunc.loss(calculatedActivations[i], expected[i]); // literal unweighted aggregate
		}
		checkRep();
		return loss;
	}

	/**
	 * Calculates and returns a gradient representing dL/dw, or the gradient of the loss (error) with respect to all the
	 * weights in the net. This works by first propagating the input through the net, calculating the loss from it using
	 * the net's loss function, and then applying a propagation algorithm using the net's sigmoid prime function and the
	 * net's loss prime function. Output will be of a dimension one less than the net's dimension; output.get(n) will
	 * represent the dL/dw for each weight between layer n and layer n+1; output.get(n)[i] will represent all the weights
	 * starting at neuron i in layer n; output.get(n)[i][j] will represent the weight from the ith neuron in layer n to
	 * the jth weight in layer n+1.
	 * Input and expected vectors are assumed to be non-null; input vector dimension must match the input dimension of
	 * the net; Expected vector dimension must match the output dimension of the net.
	 *
	 * @param input         a single vector of input / training data
	 * @param expected      the expected output vector to calculate loss against
	 * @return the matrix of error derivatives with respect to weights
	 */
	private Map<Integer, double[][]> calculateWeightGradient(double[] input, double[] expected) {
		assert input != null;
		assert expected != null;
		assert input.length == inputDim;
		assert expected.length == outputDim;
		if (depth < 2) {
			return new HashMap<>(); // no weights to adjust, return empty list
		}
		propagate(input);
		Map<Integer, double[][]> dEdw = new HashMap<>(weights.size());
		Map<Integer, double[]> dEdNet = new HashMap<>(depth);
		double[][] last_dEdw = new double[activations.get(depth - 2).length][outputDim];
		double[] last_dEdNet = new double[outputDim];
		assert weights.size() == depth - 1;
		dEdw.put(depth - 2, last_dEdw);
		dEdNet.put(depth - 1, last_dEdNet);

		double[] jLayerNeurons = activations.get(depth - 1); // last layer
		double[] iLayerNeurons = activations.get(depth - 2);
		// w_ij now conceptually points from second-to-last layer to last layer
		for (int j = 0; j < outputDim; j++) {
//			double oj = jLayerNeurons[j]; // j neuron activation in output layer
//			last_dEdNet[j] = (oj - expected[j]) * oj * (1 - oj);
			last_dEdNet[j] = lossPrime.func(sigmoidFunc.func(jLayerNeurons[j]), expected[j]) * sigmoidPrime.func(jLayerNeurons[j]);
		}
		// doing this nested loop independent of the j loop eliminates costly column-major mem access
		for (int i = 0; i < iLayerNeurons.length; i++) {
			double oi = sigmoidFunc.func(iLayerNeurons[i]); // i neuron activation in second-to-last layer
			for (int j = 0; j < jLayerNeurons.length; j++) {
				last_dEdw[i][j] = last_dEdNet[j] * oi;
			}
		}

		// back prop
		int i, j;
		for (j = depth - 2; j >= 1; j--) { // perform for all inner, right hand layers
			i = j - 1; // layer left of j; input layer in last loop run
			double[][] current_dEdw = new double[activations.get(i).length][activations.get(j).length];
			double[] current_dEdNet = new double[activations.get(j).length];
			dEdw.put(i, current_dEdw);
			dEdNet.put(j, current_dEdNet);
			jLayerNeurons = activations.get(j);
			iLayerNeurons = activations.get(i);
			double[][] next_weights = weights.get(j);
			double[] next_dEdNet = dEdNet.get(j + 1);
			for (int iJ = 0; iJ < jLayerNeurons.length; iJ++) { // calculate stage of recursive derivative term
//				double[] wOutOfJ = next_weights[iJ];
				double sum = 0;
				for (int iK = 0; iK < next_dEdNet.length; iK++) {
					sum += next_weights[iJ][iK] * next_dEdNet[iK];
				}
//				double oj = jLayerNeurons[iJ];
//				current_dEdNet[iJ] = sum * oj * (1 - oj);
				current_dEdNet[iJ] = sum * sigmoidPrime.func(jLayerNeurons[iJ]);
			}
			for (int iI = 0; iI < iLayerNeurons.length; iI++) { // calculate weight derivative
				for (int iJ = 0; iJ < jLayerNeurons.length; iJ++) {
					current_dEdw[iI][iJ] = current_dEdNet[iJ] * sigmoidFunc.func(iLayerNeurons[iI]); // TODO FIX INDEXING BUG!!
				}
			}
		}
		checkRep();
		return dEdw;
	}

	/**
	 * Performs a single step of normalized gradient descent by using the given batch of input vectors mapped to expected
	 * (training) output vectors, aggregating their weight gradients calculated through back propagation, and adjusting
	 * all weights proportional to the given step size.
	 * Input vectors in the batch are all assumed to match the input dimension of the net. Expected output vectors are
	 * assumed to match output dimensions of the net. Batch is assumed to be non-null and non-empty.
	 * Step is assumed to be positive.
	 *
	 * @param batch a map of input vectors to their expected output vectors
	 * @param step  the multiplier for weight adjustments; large values will perform coarser gradient descent
	 */
	void gradientStep(Map<double[], double[]> batch, double step) {
		assert batch != null;
		assert step > 0;
		int batchSize = batch.size();
		Map<Integer, double[][]> gradientAggregate = new HashMap<>();
		for (double[] input : batch.keySet()) {
			double[] expected = batch.get(input);
			Map<Integer, double[][]> gradient = calculateWeightGradient(input, expected);
			if (gradientAggregate.isEmpty()) {
				gradientAggregate = gradient;
			} else {
				for (int i = 0; i < gradientAggregate.size(); i++) {
					double[][] aggregate = gradientAggregate.get(i);
					double[][] single = gradient.get(i);
					for (int j = 0; j < aggregate.length; j++) {
						for (int k = 0; k < aggregate[0].length; k++) {
							aggregate[j][k] += single[j][k];
						}
					}
				}
			}
		}
		// adjust
		assert weights.size() == gradientAggregate.size();
		for (int i = 0; i < gradientAggregate.size(); i++) {
			double[][] toAdjust = weights.get(i);
			double[][] adjustBy = gradientAggregate.get(i);
			for (int j = 0; j < adjustBy.length; j++) {
				for (int k = 0; k < adjustBy[0].length; k++) {
					toAdjust[j][k] -= (step * adjustBy[j][k] / batchSize); // normalize by batch size, scale by step
				}
			}
		}
		checkRep();
	}

	/**
	 * Returns the depth of the neural net.
	 *
	 * @return the depth
	 */
	@Deprecated
	int getDepth() {
		return depth;
	}

	/**
	 * Returns the input dimension of the neural net.
	 *
	 * @return the input dimension
	 */
	int getInputDim() {
		return inputDim;
	}

	/**
	 * Returns the output dimension of the neural net.
	 *
	 * @return the output dimension
	 */
	int getOutputDim() {
		return outputDim;
	}

	/**
	 * Returns the dimension for arbitrary hidden layers in the net, or -1 if there are no hidden layers.
	 *
	 * @return the hidden layer dimension, or -1 if the net has no hidden layers
	 */
	@Deprecated
	int getHiddenLayerDim() {
		return hiddenLayerDim;
	}

	/**
	 * Returns the sigmoid function of the neural net.
	 *
	 * @return the sigmoid function
	 */
	@Deprecated
	Sigmoid getSigmoidFunc() {
		return sigmoidFunc;
	}

	/**
	 * Returns the sigmoid derivative function of the neural net.
	 *
	 * @return the sigmoid derivative function
	 */
	@Deprecated
	SigmoidPrime getSigmoidPrime() {
		return sigmoidPrime;
	}

	/**
	 * Returns the loss function of the neural net.
	 *
	 * @return the loss function
	 */
	@Deprecated
	LossFunction getLossFunc() {
		return lossFunc;
	}

	/**
	 * Returns the loss function derivative of the neural net.
	 *
	 * @return the loss function derivative
	 */
	@Deprecated
	LossFunctionPrime getLossPrime() {
		return lossPrime;
	}

	/**
	 * Checks the representation invariant
	 */
	private void checkRep() {
		if (LIGHT) {
			return;
		}
		assert depth > 0 && inputDim > 0 && outputDim > 0 && hiddenLayerDim > 0;
		assert activations != null && weights != null;
		assert random != null && sigmoidFunc != null && sigmoidPrime != null && lossFunc != null && lossPrime != null;
		assert activations.size() == depth && weights.size() == depth - 1;
		if (DEBUG) { // heavy assertions
//			assert !activations.containsKey(null) && !weights.containsKey(null);
			assert activations.get(0).length == inputDim && activations.get(depth - 1).length == outputDim;
			for (int i = 0; i < weights.size(); i++) {
				assert weights.get(i).length == activations.get(i).length;
				assert weights.get(i)[0].length == activations.get(i + 1).length;
			}
			if (depth > 2) {
				for (int i = 1; i < depth - 1; i++) {
					assert activations.get(i).length == hiddenLayerDim;
				}
			}
		}
	}

}
