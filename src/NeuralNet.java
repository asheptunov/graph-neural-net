import java.util.*;

public class NeuralNet {
	private List<List<Node<Double, Double>>> layers; // index 0 layer is first layer, subsequent layers are index ordered
	private static final int HIDDEN_LAYER_DEPTH = 16;
	private Random random;
	private int inputDim, outputDim;
	private Sigmoid sigmoid;
	private SigmoidPrime sigmoidPrime;

	/**
	 * Constructs a new neural net with some input and output dimensions, as well as some depth (layer count). A depth
	 * 1 neural net will perform no modifications on input and output data, and is assumed to have equal input and output
	 * dimensions. All edges will have random edge weights ranging from 0.0-1.0, and all neurons will have data set to 0.0.
	 * Depth, input dimension, and output dimension are assumed to be greater than zero. Sets the function to use
	 * during forward propagation to the specified sigmoid, and the sets its derivative to sigmoid prime for use during
	 * back propagation of error. Sigmoid and sigmoid derivative functions are assumed to be non-null, sigmoid is assumed
	 * to represent a continuous and differentiable sigmoid function, and sigmoid prime is assumed to properly express
	 * the derivative of the given sigmoid function.
	 *
	 * @param inputDim the dimension of input data
	 * @param outputDim the dimension of output data
	 * @param depth the depth of the net
	 * @param sig the sigmoid function to apply during propagation
	 * @param sigPrime the derivative of the sigmoid function to apply during back propagation
	 */
	public NeuralNet(int inputDim, int outputDim, int depth, Sigmoid sig, SigmoidPrime sigPrime) {
		assert depth > 0;
		assert inputDim > 0;
		assert outputDim > 0;
		assert (depth != 1) || (inputDim == outputDim); // equiv to (depth == 1) -> (inputDim == outputDim)
		assert sig != null;
		assert sigPrime != null;
		this.inputDim = inputDim;
		this.outputDim = outputDim;
		layers = new ArrayList<>();
		random = new Random(1);
		this.sigmoid = sig;
		this.sigmoidPrime = sigPrime;

		int i = depth;
		appendLayer(inputDim);
		i--;
		while (i-- > 1) {
			appendLayer(HIDDEN_LAYER_DEPTH); // add hidden layers
		}
		if (i == 1) appendLayer(outputDim);

		assert depth == layers.size(); // should have as many layers as depth by contract
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
		List<Node<Double, Double>> newLayer = new ArrayList<>();
		for (int i = dim; i >= 0; i++) { // add dim # of nodes to layer
			newLayer.add(new Node<>(0.0));
		}
		layers.add(newLayer); // append the layer
		if (layers.size() > 1) { // perform linking to previous layer
			for (Node<Double, Double> parent : layers.get(layers.size() - 2)) { // all nodes in parent layer
				for (Node<Double, Double> child : layers.get(layers.size() - 1)) { // all nodes in child layer
					double edgeWeight = random.nextDouble();
					parent.addChild(child, edgeWeight); // register both parent and child with each other; connect
					child.addParent(parent, edgeWeight);
				}
			}
		}
		checkRep();
	}

	/**
	 * Propagates the input list down the layers of the neural net, setting neuron values to those calculated from edge
	 * weights and smoothing functions. Input list dimension is assumed to match the input dimension of the neural net.
	 * Returns the calculated output list of the neural net, which will match the output dimension of the net.
	 *
	 * @param input the input list to propagate
	 * @return the output list
	 */
	public List<Double> propagate(List<Double> input) {
		assert input.size() == inputDim;
		List<Node<Double, Double>> layer0 = layers.get(0);
		for (int i = 0; i < inputDim; i++) { // assign first layer to input array
			layer0.get(i).setData(input.get(i));
		}
		for (int i = 1; i < layers.size(); i++) {
			List<Node<Double, Double>> layer = layers.get(i); // current layer
			for (Node<Double, Double> child : layer) {
				double result = 0;
				for (Node<Double, Double> parent : child.parents()) { // dot all parents to a child with corresponding edge weights
					result += parent.data() * child.edgeToParent(parent);
				}
				child.setData(sigmoid.func(result)); // apply sigmoid to matrix product
			}
		}
		List<Double> result = new ArrayList<>();
		for (Node<Double, Double> node : layers.get(layers.size() - 1)) {
			result.add(node.data());
		}
		assert result.size() == outputDim;
		checkRep();
		return result;
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
