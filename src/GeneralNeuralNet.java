import java.util.*;

/**
 * Represents a neural network with some type of neurons, capable of propagating input, back propagating error, and
 * performing batch gradient descent to train the network.
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
public class GeneralNeuralNet implements NeuralNet {
	// structure
	private int layers; // # of neuron layers
	private int inputDim, outputDim; // # of neurons in input and output layer, respectively
    private int[] layerDims;
	private Map<Integer, double[]> neurons; // neuron sums (unactivated values)
	private Map<Integer, double[][]> weights; // weights

	// functions for math
	private Random random;
	private ActivationFunction activationFunc; // activation function for all but last layer
	private ActivationPrime activationPrime; // activation derivative for all but last layer
	private ActivationFunction lastActivationFunc; // activation function for last layer
	private ActivationPrime lastActivationPrime; // activation derivative for last layer
	private LossFunction lossFunc;
	private LossFunctionPrime lossPrime;

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
	// neurons, weights, random, activationFunc, activationPrime, lossFunc, lossPrime != null
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
	 * Constructs a new neural net with some input, output, and hidden layer dimensions, and a given layer count.
	 * A layers 1 neural net will perform no modifications on input and output data, and is assumed to have equal
	 * input and output dimensions. All edges will have random edge weights ranging from 0.0-1.0, and all neurons will
	 * have data set to 0.0. If hidden layers exist, all will have the same specified dimension. All layers that aren't
	 * the input or output layer will be hidden layers. Sets the function to use during forward propagation to the
	 * specified sigmoid, and the sets its derivative to sigmoid prime for use during back propagation of error. Does the
	 * same for a given loss function and its derivative, which will be used during error calculation and back propagation,
	 * respectively.
	 * Depth, input dimension, and output dimension are assumed to be greater than zero.
	 * {@code activationFunc} and {@code activationPrime} are assumed to be non-null, {@code activationPrime} is assumed
	 * to represent a continuous and differentiable function, and {@code activationPrime} is assumed to correctly express
	 * the derivative of the given activation function. The same criteria are assumed for the {@code lastActivationFunc}
	 * and {@code lastActivationPrime}, as well as for {@code lossFunc} and {@code lossPrime}.
     * todo hidden dim description
     *
     * todo require at least 2 layers total
	 *
	 * @param activationFunc the activation function to apply to all but the last layer during propagation
	 * @param activationPrime the derivative of the activation function to apply to all but the last layer during back propagation
	 * @param lastActivationFunc the activation function to apply to the last layer during propagation
	 * @param lastActivationPrime the derivative of the activation function to apply to the last layer during back propagation
	 * @param lossFunc the loss function to apply during error evaluation
	 * @param lossPrime the derivative of the loss function to apply during back propagation
	 */
	public GeneralNeuralNet(int[] layerDims,
					 ActivationFunction activationFunc, ActivationPrime activationPrime,
					 ActivationFunction lastActivationFunc, ActivationPrime lastActivationPrime,
					 LossFunction lossFunc, LossFunctionPrime lossPrime) {

		// precondition checks
		assert layerDims != null && layerDims.length >= 2;
		assert activationFunc != null && activationPrime != null;
		assert lastActivationFunc != null && lastActivationPrime != null;
		assert lossFunc != null && lossPrime != null;

		// function init
		this.activationFunc = activationFunc;
		this.activationPrime = activationPrime;
		this.lastActivationFunc = lastActivationFunc;
		this.lastActivationPrime = lastActivationPrime;
		this.lossFunc = lossFunc;
		this.lossPrime = lossPrime;
		random = new Random(1);

		// structure init
		this.layers = layerDims.length;
		this.inputDim = layerDims[0];
		this.outputDim = layerDims[layers - 1];
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
	 * @param dim the dimension of the layer to construct
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
	 * Propagates the input vector down the layers of the neural net, and returns the output after propagation.
	 * Input vector is assumed to be non-null, and its dimension is assumed to match the net's input dimension.
	 *
	 * @param input the input vector to propagate
	 * @return the output vector
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
				previousActivations[i] = activationFunc.func(previousNeurons[i]);
			}
			for (int i = 0; i < previousActivations.length; i++) {
				for (int j = 0; j < currentNeurons.length; j++) {
					// propagates one value from previous layer into all values from next layer
					currentNeurons[j] += previousActivations[i] * intermediateWeights[i][j];
				}
			}

		}
		double[] outputNeurons = neurons.get(layers - 1);
		double[] outputActivations = new double[outputDim];
		for (int i = 0; i < outputDim; i++) {
			outputActivations[i] = lastActivationFunc.func(outputNeurons[i]);
		}
		checkRep();
		return outputActivations;
	}

	/**
	 * Calculates and returns a loss for a given input set by forward propagating the input, comparing it to the
	 * provided expected output and applying the net's loss function per output component, and aggregating the components.
	 * Assumes {@code input} is non-null and has same dimension as the net's input dimension.
	 * Assumes the {@code expected} is non-null and has the same dimension as the net's output dimension.
	 *
	 * @param input    a single vector of input data
	 * @param expected the corresponding expected output vector
	 * @return the calculated loss vector
	 */
	public double calculateLoss(double[] input, double[] expected) {
		// precondition checks
		assert input != null && expected != null;
		assert input.length == inputDim && expected.length == outputDim;

		double[] calculatedActivations = propagate(input);
		double loss = 0;
		for (int i = 0; i < outputDim; i++) {
			loss += lossFunc.loss(calculatedActivations[i], expected[i]);
		}
		checkRep();
		return loss;
	}

	/**
	 * Calculates and returns a gradient representing dL/dw, or the gradient of the loss (error) with respect to all the
	 * weights in the net. Output list will be of a dimension one less than the net's dimension; output.get(n) will
	 * represent the dL/dw for each weight between layer n and layer n+1 in the net; output.get(n)[i] will represent all
	 * the weights starting at neuron i in layer n; output.get(n)[i][j] will represent the weight from the ith neuron in
	 * layer n to the jth weight in layer n+1.
	 * {@code input} and {@code expected} are assumed to be non-null.
	 * {@code input} dimension must match the input dimension of the net; {@code expected} dimension must match the output
	 * dimension of the net.
	 *
	 * @param input         a single vector of input / training data
	 * @param expected      the corresponding expected output vector
	 * @return a matrix of weight derivatives
	 */
	public Map<Integer, double[][]> calculateWeightGradient(double[] input, double[] expected) {
		// precondition checks
		assert input != null && expected != null;
		assert input.length == inputDim && expected.length == outputDim;

		if (layers < 2) {
			return new HashMap<>(); // no weights to adjust, return empty list
		}
		propagate(input); // neurons now represent sums before activation
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
			double jActivation = lastActivationFunc.func(jNeurons[j]); // use last layer activation function since j is outer layer
			last_dEdNet[j] = lossPrime.func(jActivation, expected[j]) * lastActivationPrime.func(jNeurons[j]); // use last layer prime
		}
		// doing this nested loop independent of the j loop eliminates costly column-major mem access
		for (int i = 0; i < iNeurons.length; i++) {
			double iActivation = activationFunc.func(iNeurons[i]); // use normal activation function since i is not last layer
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
				dEdNet_j[j] = sum * activationPrime.func(jNeurons[j]);
			}
			for (int i = 0; i < iNeurons.length; i++) {
				for (int j = 0; j < jNeurons.length; j++) {
					dEdw_ij[i][j] = dEdNet_j[j] * activationFunc.func(iNeurons[i]);
				}
			}
		}
		checkRep();
		return dEdw;
	}

	/**
	 * Performs a single step of normalized gradient descent using the given batch of input -> expected output vector
	 * mappings. Weight gradients calculated for each mapping will be aggregated and normalized by the batch size, and
	 * all weights will be adjusted directly proportional to the given step size. Large {@code step} values will perform
	 * coarser gradient descent and large quantities of gradient steps with large step size may cause the net to diverge.
	 * Input vectors in {@code batch} are all assumed to match the input dimension of the net.
	 * Expected output vectors in {@code batch} are assumed to match output dimensions of the net.
	 * {@code batch} is assumed to be non-null and non-empty.
	 * {@code step} is assumed to be positive.
	 * todo add momentum term and noise description
	 *
	 * @param batch a map of input vectors to their expected output vectors
	 * @param step  the multiplier for weight adjustments
	 * @param momentum the momentum term
	 * @param noise whether or not to apply gaussian noise
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
	 * Checks the representation invariant
	 */
	private void checkRep() {
		if (LIGHT) return;

		// todo update checkrep, rep invariant, and abs func to match new optimizations

        assert layers > 0 && inputDim > 0 && outputDim > 0;
        // basic assertions
		assert neurons != null && weights != null;
		assert random != null && activationFunc != null && activationPrime != null && lossFunc != null && lossPrime != null;
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
