import java.util.*;

/**
 * Represents a neural network with sigmoid neurons, capable of propagating input, back propagating error, and performing
 * batch gradient descent to train the network.
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
public class NeuralNet {
	// structure
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

	// whether or not to run heavy rep inv checks
	private boolean debug = true;
	// whether or not to run rep inv checks at all
	private boolean light = false;

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
		this.hiddenLayerDim = (depth > 2) ? hiddenLayerDim : -1; // hidden dim is -1 if no hidden layers
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
	 * weights and smoothing functions. Returns the output vector after propagation.
	 * Input vector is assumed to be non-null, and its dimension is assumed to match the net's input dimension.
	 *
	 * @param input the input vector to propagate
	 * @return the output vector
	 */
	public double[] propagate(double[] input) {
		assert input != null;
		assert input.length == inputDim;
		System.arraycopy(input, 0, activations.get(0), 0, inputDim); // copy into first layer
		for (int l = 1; l < depth; l++) { // traverse through output layer multiplying layer vector by weights matrix
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
		double[] output = new double[outputDim];
		System.arraycopy(activations.get(depth - 1), 0, output, 0, outputDim); // copy out last layer
		checkRep();
		return output;
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
	public double calculateLoss(double[] input, double[] expected) {
		assert input != null;
		assert expected != null;
		assert input.length == inputDim;
		assert expected.length == outputDim;
		double[] calculated = propagate(input);
		double output = 0;
		for (int i = 0; i < outputDim; i++) {
			output += lossFunc.loss(calculated[i], expected[i]);
		}
		checkRep();
		return output;
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
	public List<double[][]> calculateWeightGradient(double[] input, double[] expected) {
		assert input != null;
		assert expected != null;
		assert input.length == inputDim;
		assert expected.length == outputDim;
		if (depth < 2) {
			return new ArrayList<>(); // no weights to adjust, return empty list
		}
		List<double[][]> dEdw = new ArrayList<>(weights.size());
		List<double[]> dEdNet = new ArrayList<>(depth);
		double[][] last_dEdw = new double[activations.get(depth - 2).length][outputDim];
		double[] last_dEdNet = new double[outputDim];
		assert weights.size() == depth - 1;
		dEdw.set(depth - 2, last_dEdw);
		dEdNet.set(depth - 1, last_dEdNet);

		double[] jLayer = activations.get(depth - 1); // last layer
		double[] iLayer = activations.get(depth - 2);
		// w_ij now conceptually points from second-to-last layer to last layer
		for (int j = 0; j < outputDim; j++) {
			double oj = jLayer[j];
			last_dEdNet[j] = (oj - expected[j]) * oj * (1 - oj);
		}
		// doing this nested loop independent of the j loop eliminates costly column-major mem access
		for (int i = 0; i < iLayer.length; i++) {
			double oi = iLayer[i];
			for (int j = 0; j < jLayer.length; j++) {
				last_dEdw[i][j] = last_dEdNet[j] * oi;
			}
		}

		// back prop
		int i, j;
		for (j = depth - 2; j >= 1; j--) { // perform for all inner, right hand layers
			i = j - 1;
			double[][] current_dEdw = new double[activations.get(i).length][activations.get(j).length];
			double[] current_dEdNet = new double[activations.get(j).length];
			dEdw.set(i, current_dEdw);
			dEdNet.set(j, current_dEdNet);
			jLayer = activations.get(j);
			iLayer = activations.get(i);
			double[][] next_weights = weights.get(j);
			double[] next_dEdNet = dEdNet.get(j + 1);
			for (int iJ = 0; iJ < jLayer.length; iJ++) { // calculate stage of recursive derivative term
				double[] wOutOfJ = next_weights[iJ];
				int sum = 0;
				for (int iNext = 0; iNext < next_dEdNet.length; iNext++) {
					sum += next_weights[iJ][iNext] * next_dEdNet[iNext];
				}
				double oj = jLayer[iJ];
				current_dEdNet[iJ] = sum * oj * (1 - oj);
			}
			for (int iI = 0; iI < iLayer.length; iI++) { // calculate weight derivative
				for (int iJ = 0; iJ < jLayer.length; iJ++) {
					current_dEdw[iI][iJ] = current_dEdNet[j] * iLayer[iI];
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
	public void gradientStep(Map<double[], double[]> batch, double step) {
		assert batch != null;
		assert step > 0;
		int batchSize = batch.size();
		List<double[][]> gradientAggregate = new ArrayList<>();
		for (double[] input : batch.keySet()) {
			double[] expected = batch.get(input);
			List<double[][]> gradient = calculateWeightGradient(input, expected);
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
		for (int i = 0; i < gradientAggregate.size(); i++) {
			double[][] toAdjust = weights.get(i);
			double[][] adjustBy = gradientAggregate.get(i);
			for (int j = 0; j < adjustBy.length; j++) {
				for (int k = 0; k < adjustBy[0].length; k++) {
					toAdjust[i][j] -= (step * adjustBy[i][j] / batchSize); // normalize by batch size, scale by step
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
	public int getDepth() {
		return depth;
	}

	/**
	 * Returns the input dimension of the neural net.
	 *
	 * @return the input dimension
	 */
	public int getInputDim() {
		return inputDim;
	}

	/**
	 * Returns the output dimension of the neural net.
	 *
	 * @return the output dimension
	 */
	public int getOutputDim() {
		return outputDim;
	}

	/**
	 * Returns the dimension for arbitrary hidden layers in the net, or -1 if there are no hidden layers.
	 *
	 * @return the hidden layer dimension, or -1 if the net has no hidden layers
	 */
	public int getHiddenLayerDim() {
		return hiddenLayerDim;
	}

	/**
	 * Returns the sigmoid function of the neural net.
	 *
	 * @return the sigmoid function
	 */
	public Sigmoid getSigmoidFunc() {
		return sigmoidFunc;
	}

	/**
	 * Returns the sigmoid derivative function of the neural net.
	 *
	 * @return the sigmoid derivative function
	 */
	public SigmoidPrime getSigmoidPrime() {
		return sigmoidPrime;
	}

	/**
	 * Returns the loss function of the neural net.
	 *
	 * @return the loss function
	 */
	public LossFunction getLossFunc() {
		return lossFunc;
	}

	/**
	 * Returns the loss function derivative of the neural net.
	 *
	 * @return the loss function derivative
	 */
	public LossFunctionPrime getLossPrime() {
		return lossPrime;
	}

	/**
	 * Checks the representation invariant
	 */
	private void checkRep() {
		if (light) {
			return;
		}
		assert depth > 0 && inputDim > 0 && outputDim > 0 && hiddenLayerDim > 0;
		assert activations != null && weights != null;
		assert random != null && sigmoidFunc != null && sigmoidPrime != null && lossFunc != null && lossPrime != null;
		assert activations.size() == depth && weights.size() == depth - 1;
		if (debug) { // heavy assertions
			assert !activations.contains(null) && !weights.contains(null);
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
