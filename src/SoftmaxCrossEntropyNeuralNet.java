import java.util.*;

/**
 * Represents a neural network with configurable inner neuron activation functions, Softmax activation on the outer layer,
 * and a cross-entropy loss function. This neural net is capable of propagating input, back propagating error, and
 * performing mini-batch gradient descent to train.
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
public class SoftmaxCrossEntropyNeuralNet implements NeuralNet {
	// structure
	private int layers; // # of neuron layers
	private int inputDim, outputDim; // # of neurons in input and output layer, respectively
	private int[] layerDims;
	private Map<Integer, double[]> neurons; // neuron sums (unactivated values)
	private Map<Integer, double[][]> weights; // weights

	// functions for math
	private Random random;
	private ActivationFunction innerActivationFunc; // activation function for all but last layer
	private ActivationPrime innerActivationPrime; // activation derivative for all but last layer

	// training optimizers
	private int time; // the number of times a gradient step has been taken
	private Map<Integer, double[][]> previousUpdate; // update vector used in the previous time step,
	// or vector of 0's matching weights map dimensions, if the net has not undergone gradient descent

	// whether or not to run heavy rep inv checks
	private static final boolean DEBUG = false;
	// whether or not to run rep inv checks at all
	private static final boolean LIGHT = true;
	// whether or not to trust client not to destroy rep invariant in order to gain speed;
	// DO NOT SET THIS TO TRUE UNLESS YOU ARE EXTREMELY CONFIDENT THAT YOU HAVE NOT TAMPERED WITH THE REP INVARIANT
	private static final boolean TRUST = true;

	// Abstraction function:
	// A GeneralNeuralNet represents the ADT of a neural network with layers # of layers and a layers - 1 # of weight layers.
	// These correspond to the lengths of arrays neurons and weights, respectively. The net has an input
	// dimension of inputDim, and an output dimension of outputDim. These correspond to the lengths of the first
	// and last elements in array neurons, respectively.
	// Array neurons[i] refers to the vector of neuron neurons in layer i.
	// Value neurons[i][j] refers to the value of the jth neuron in layer i.
	// Matrix weights[i] refers to the matrix of all edge weights between the ith and i+1th activation layers.
	// Array weights[i][j] refers to the vector of edge weights originating from the jth neuron in the ith activation layer.
	// Value weights[i][j][k] refers to the value of the edge weight from the jth neuron in the ith activation layer
	// to the kth neuron in the i+1th activation layer.
	//
	// Examples:
	// A layers 1 neural net will have matching inputDim and outputDim, have a single element in neurons, and an
	// empty weights array. It will have no hidden layers.
	// A layers 2 neural net will have 2 elements in neurons, and 1 element in weights. It will have no hidden layers.
	// A layers 3 neural net will have 3 elements in neurons, and 2 elements in weights. It will have 1 hidden layer
	// of dimension hiddenLayerDim.

	// Representation invariant:
	// layers, inputDim, outputDim, hiddenLayerDim > 0
	// neurons, weights, random, innerActivationFunc, innerActivationPrime, lossFunc, lossPrime != null
	// neurons[...], weights[...], weights[...][...] != null
	// neurons.length == layers
	// weights.length == layers - 1
	// neurons[0].length == inputDim
	// neurons[layers - 1].length == outputDim
	// weights[i].length == neurons[i].length
	// weights[i][j].length == neurons[i + 1].length
	// layers > 2 -> neurons[1...(layers - 2)].length == hiddenLayerDim

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
	// neurons[0] weights[0] neurons[1]   neurons[layers-1]
	//

	/**
	 * Constructs a new neural net from some dimension vector, which will correspond in order with the resulting layer
	 * dimensions. Layers will be fully connected, and have random edge weights ranging from 0.0-1.0. All neurons will
	 * have data set to 0.0. Sets the specified activation function to use during forward propagation on the inner layers,
	 * and uses a Softmax activation function on the last layer. Sets the derivative of inner activation function as
	 * specified. Uses a cross-entropy loss function.
	 * Layer dimension vector is assumed to be non-null, non-empty, and contain at least 2 positive entries.
	 * Inner activation function and its derivative are assumed to be non-null and continuous. Inner activation function
	 * assumed to be differentiable, and its derivative is assumed to be correctly expressed.
	 *
	 * @param layerDims the dimension vector for layers in the net
	 * @param innerActivationFunc the activation function for inner layers
	 * @param innerActivationPrime the derivative of the inner activation function
	 */
	SoftmaxCrossEntropyNeuralNet(int[] layerDims, ActivationFunction innerActivationFunc, ActivationPrime innerActivationPrime) {
		// non-null checks
		assert layerDims != null && layerDims.length >= 2;
		assert innerActivationFunc != null && innerActivationPrime != null;

		// function init
		this.innerActivationFunc = innerActivationFunc;
		this.innerActivationPrime = innerActivationPrime;
		random = new Random(1); // seed for repeatability

		// structure init
		this.layers = layerDims.length;
		this.inputDim = layerDims[0];
		this.outputDim = layerDims[layerDims.length - 1];
		this.layerDims = layerDims;
		neurons = new HashMap<>();
		weights = new HashMap<>();

		// helper init
		time = 0;
		previousUpdate = new HashMap<>();

		// build
		for (int i = 0; i < layers; i++) {
			appendLayer(i, layerDims[i]);
		}

		checkRep();
	}

	/**
	 * Appends a new layer of the specified number of neurons to the neural net, and links all new nodes with all the
	 * nodes from the previous layer (if such exists) using a cartesian product of edges. New edges will have random edge
	 * weights ranging from -0.5 (inclusive) to 0.5 (exclusive), and new neurons will be set to 0.0.
	 * Assumes the input dimension is greater than zero, and the layerIndex is valid for the net's dimensions.
	 *
	 * @param layerIndex the index of the layer in the network
	 * @param dim        the dimension of the layer to construct
	 */
	private void appendLayer(int layerIndex, int dim) {
		// precondition checks
		assert layerIndex >= 0 && layerIndex < layers;
		assert dim > 0;

		double[] newNeurons = new double[dim]; // all values automatically 0.0
		if (layerIndex > 0) { // we are adding a hidden layer or output layer; add weights
			double[][] newWeights = new double[neurons.get(layerIndex - 1).length][dim];
			for (int i = 0; i < newWeights.length; i++) {
				for (int j = 0; j < dim; j++) {
					newWeights[i][j] = random.nextDouble() - 0.5; // -.5 to .5
				}
			}
			weights.put(layerIndex - 1, newWeights);
			previousUpdate.put(layerIndex - 1, new double[neurons.get(layerIndex - 1).length][dim]); // array of 0's
		}
		neurons.put(layerIndex, newNeurons);
	}

	/**
	 * {@inheritDoc}
	 *
	 * @param input the input vector
	 * @return {@inheritDoc}
	 */
	public double[] propagate(double[] input) {
		assert input != null;
		assert input.length == inputDim;
		if (TRUST) {
			neurons.put(0, input); // O(n) -> constant if trust client
		} else {
			System.arraycopy(input, 0, neurons.get(0), 0, inputDim); // copy into first layer
		}
		for (int l = 1; l < layers; l++) { // traverse through output layer multiplying layer vector by weights matrix
			double[] previousNeurons = neurons.get(l - 1); // l-1th layer neurons (previous)
			double[] currentNeurons = neurons.get(l); // lth layer neurons (current)
			double[][] intermediateWeights = weights.get(l - 1); // weights between l-1th layer (previous) and lth layer (current)

			assert intermediateWeights.length == previousNeurons.length;
			assert intermediateWeights[0].length == currentNeurons.length;

			for (int i = 0; i < currentNeurons.length; i++) {
				currentNeurons[i] = 0.0; // reset all values
			}
			double[] previousActivations = new double[previousNeurons.length]; // save time by calculating activations once
			for (int i = 0; i < previousNeurons.length; i++) {
				previousActivations[i] = innerActivationFunc.func(previousNeurons[i]);
			}
			for (int i = 0; i < previousActivations.length; i++) {
				for (int j = 0; j < currentNeurons.length; j++) {
					// propagates one value from previous layer into all values from next layer
					currentNeurons[j] += previousActivations[i] * intermediateWeights[i][j];
				}
			}

		}
		double[] outputNeurons = neurons.get(layers - 1);
		checkRep();
		return softmaxActivate(outputNeurons);
	}

	/**
	 * {@inheritDoc}
	 *
	 * @param input    the input vector
	 * @param expected the corresponding expected output vector
	 * @return {@inheritDoc}
	 */
	public double calculateLoss(double[] input, double[] expected) {
		// precondition checks
		assert input != null && expected != null;
		assert input.length == inputDim && expected.length == outputDim;

		double[] calculatedActivations = propagate(input);
		double loss = 0;
		for (int i = 0; i < outputDim; i++) {
			assert calculatedActivations[i] > 0.0; // this would cause log problems, and shouldn't happen with softmax
			loss -= expected[i] * Math.log(calculatedActivations[i]);
		}
		checkRep();
		return loss;
	}

	/**
	 * {@inheritDoc}
	 *
	 * @param input    the input vector
	 * @param expected the corresponding expected output vector
	 * @return {@inheritDoc}
	 */
	public Map<Integer, double[][]> calculateWeightGradient(double[] input, double[] expected) {
		// precondition checks
		assert input != null && expected != null;
		assert input.length == inputDim && expected.length == outputDim;

		if (layers < 2) {
			return new HashMap<>(); // no weights to adjust, return empty list
		}
		double[] outputActivations = propagate(input);
		Map<Integer, double[][]> dEdw = new HashMap<>(weights.size());
		Map<Integer, double[]> dEdNet = new HashMap<>(layers);
		double[][] last_dEdw = new double[neurons.get(layers - 2).length][outputDim];
		double[] last_dEdNet = new double[outputDim];
		dEdw.put(layers - 2, last_dEdw);
		dEdNet.put(layers - 1, last_dEdNet);

		// OUTPUT LAYER CASE
		double[] jNeurons = neurons.get(layers - 1); // last layer
		double[] iNeurons = neurons.get(layers - 2); // second-to-last layer
		// w_ij now conceptually points from second-to-last layer to last layer
		for (int j = 0; j < outputDim; j++) {
			last_dEdNet[j] = outputActivations[j] - expected[j]; // this is unique for softmax + cross-entropy combo
		}
		// doing this nested loop independent of the j loop eliminates costly column-major mem access
		for (int i = 0; i < iNeurons.length; i++) {
			double iActivation = innerActivationFunc.func(iNeurons[i]); // use normal activation function since i is not last layer
			for (int j = 0; j < jNeurons.length; j++) {
				last_dEdw[i][j] = last_dEdNet[j] * iActivation;
			}
		}

		// INNER LAYER CASE
		int iLayer, jLayer;
		for (jLayer = layers - 2; jLayer >= 1; jLayer--) { // perform for all inner, right hand layers
			iLayer = jLayer - 1; // layer left of j; input layer in last loop run
			double[][] dEdw_ij = new double[neurons.get(iLayer).length][neurons.get(jLayer).length];
			double[] dEdNet_j = new double[neurons.get(jLayer).length];
			dEdw.put(iLayer, dEdw_ij);
			dEdNet.put(jLayer, dEdNet_j);
			jNeurons = neurons.get(jLayer);
			iNeurons = neurons.get(iLayer);
			double[][] weights_jk = weights.get(jLayer); // weights between j and j+1 layers
			double[] dEdNet_k = dEdNet.get(jLayer + 1); // dEdNet for j+1 layer
			for (int j = 0; j < jNeurons.length; j++) {
				double sum = 0;
				for (int k = 0; k < dEdNet_k.length; k++) {
					sum += weights_jk[j][k] * dEdNet_k[k];
				}
				dEdNet_j[j] = sum * innerActivationPrime.func(jNeurons[j]);
			}
			for (int i = 0; i < iNeurons.length; i++) {
				for (int j = 0; j < jNeurons.length; j++) {
					dEdw_ij[i][j] = dEdNet_j[j] * innerActivationFunc.func(iNeurons[i]);
				}
			}
		}
		checkRep();
		return dEdw;
	}

	/**
	 * {@inheritDoc}
	 *
	 * @param batch    the batch of input vectors mapped to expected output vectors
	 * @param step     the step size; higher values will make coarser updates
	 * @param momentum the fraction of the previous gradient step to add
	 * @param noise    whether or not to add gaussian noise to weight updates
	 */
	public void gradientStep(Map<double[], double[]> batch, double step, double momentum, boolean noise) {
		// precondition checks
		assert batch != null;
		assert !batch.isEmpty();
		assert step > 0;

		// calculate and aggregate gradients
		Map<Integer, double[][]> gradientAggregate = new HashMap<>();
		for (double[] input : batch.keySet()) {
			double[] expected = batch.get(input);
			Map<Integer, double[][]> gradient = calculateWeightGradient(input, expected);
			if (!gradientAggregate.isEmpty()) {
				for (int i = 0; i < gradientAggregate.size(); i++) {
					double[][] aggregate = gradientAggregate.get(i);
					double[][] single = gradient.get(i);
					for (int j = 0; j < aggregate.length; j++) {
						for (int k = 0; k < aggregate[0].length; k++) {
							aggregate[j][k] += single[j][k];
						}
					}
				}
			} else {
				gradientAggregate = gradient;
				// first gradient calculation, simply assign in which will set dimensions and base for aggregation
			}
		}

		// adjust
		int batchSize = batch.size();
		assert weights.size() == gradientAggregate.size();
		for (int i = 0; i < gradientAggregate.size(); i++) {
			double[][] toAdjust = weights.get(i);
			double[][] adjustBy = gradientAggregate.get(i);
			double[][] prevAdjust = previousUpdate.get(i);
			for (int j = 0; j < adjustBy.length; j++) {
				for (int k = 0; k < adjustBy[0].length; k++) {
					double adj = step * adjustBy[j][k] / batchSize + momentum * prevAdjust[j][k];
					if (noise) {
						adj += Math.sqrt(random.nextGaussian() * step / Math.pow(1 + time, momentum));
					}
					// normalize by batch size, scale by step
					toAdjust[j][k] -= adj;
					adjustBy[j][k] = adj; // store adjustment for learning optimization in next step
				}
			}
		}
		time++;
		previousUpdate = gradientAggregate;
		checkRep();
	}


	/**
	 * Activates the given input vector using the max-normalized Softmax function, interpreting the layer as an unactivated
	 * layer of competing neurons in the net. Returns the calculated output vector.
	 * Input vector is assumed to be non-null and non-empty.
	 *
	 * @param input the input vector
	 * @return the output vector
	 */
	private double[] softmaxActivate(double[] input) {
		assert input != null;
		int l = input.length;
		assert l > 0;
		double max = input[0];
		for (int i = 1; i < l; i++) {
			if (input[i] > max) max = input[i];
		}
		double sum = 0;
		double[] output = new double[l];
		for (int i = 0; i < l; i++) {
			double term = Math.exp(input[i]);
			output[i] = term;
			sum += term;
		}
		for (int i = 0; i < l; i++) {
			output[i] /= sum;
		}
		return output;
	}

	/**
	 * {@inheritDoc}
	 *
	 * @return {@inheritDoc}
	 */
	public int getInputDim() {
		return inputDim;
	}

	/**
	 * {@inheritDoc}
	 *
	 * @return {@inheritDoc}
	 */
	public int getOutputDim() {
		return outputDim;
	}

	/**
	 * Checks the representation invariant
	 */
	private void checkRep() {
		if (LIGHT) return;

		// todo update checkrep, rep invariant, and abs func to match new optimizations

		assert layers > 0 && inputDim > 0 && outputDim > 0;
		// basic assertions
		assert neurons != null && weights != null;
		assert random != null && innerActivationFunc != null && innerActivationPrime != null;
		assert neurons.size() == layers && weights.size() == layers - 1;
		assert neurons.get(0).length == inputDim && neurons.get(layers - 1).length == outputDim;

		// heavy assertions
		if (DEBUG) {
			for (int i = 0; i < weights.size(); i++) {
				assert weights.get(i).length == neurons.get(i).length;
				assert weights.get(i)[0].length == neurons.get(i + 1).length;
			}
			for (int i = 0; i < layers; i++) {
				assert neurons.get(i).length == layerDims[i];
			}
		}
	}

}
