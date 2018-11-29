import java.util.Map;

/**
 * Represents a general neural network capable of propagating input vectors, calculating the loss for propagated input,
 * and performing iterative mini-batch gradient descent to train.
 */
interface NeuralNet {

	/**
	 * Propagates the input vector through the net, and returns the output vector.
	 * Input vector must be non-null and match {@link #getInputDim()}.
	 *
	 * @param input the input vector
	 * @return the output vector
	 */
    double[] propagate(double[] input);

	/**
	 * Propagates the input vector, calculates its loss against the expected output vector, and returns the loss.
	 * Input vector must be non-null and match {@link #getInputDim()}. Output vector must be non-null and match
	 * {@link #getOutputDim()}.
	 *
	 * @param input    the input vector
	 * @param expected the corresponding expected output vector
	 * @return the loss
	 */
	double calculateLoss(double[] input, double[] expected);

	/**
	 * Calculates the weight gradient for the given input vector and corresponding expected output vector, and returns
	 * it. The ith entry in the gradient map will refer to the matrix of weights between the ith and i+1th layers in the
	 * net. The corresponding matrices will have as many rows as the ith layer has neurons, and as many columns as the
	 * i+1th layer has neurons.
	 *
	 * @param input    the input vector
	 * @param expected the corresponding expected output vector
	 * @return the weight gradient
	 */
    Map<Integer, double[][]> calculateWeightGradient(double[] input, double[] expected);

	/**
	 * Calculates the gradient for an entire mini-batch of input vectors mapped to expected output vectors, normalizes
	 * it against the batch size, and takes a step in the given direction, adding the specified fraction of the previous
	 * gradient step to simulate training momentum. If specified, adds gaussian noise to the weight updates.
	 *
	 * @param batch    the batch of input vectors mapped to expected output vectors
	 * @param step     the step size; higher values will make coarser updates
	 * @param momentum the fraction of the previous gradient step to add
	 * @param noise    whether or not to add gaussian noise to weight updates
	 */
    void gradientStep(Map<double[], double[]> batch, double step, double momentum, boolean noise);

	/**
	 * Returns the input dimension of the neural net.
	 *
	 * @return the input dimension
	 */
	int getInputDim();

	/**
	 * Returns the output dimension of the neural net.
	 *
	 * @return the output dimension
	 */
    int getOutputDim();

}
