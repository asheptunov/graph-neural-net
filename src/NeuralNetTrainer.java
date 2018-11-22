import java.io.OutputStream;
import java.io.PrintStream;
import java.util.*;

/**
 * Represents a machine learning module that trains a neural network by performing iterative stochastic batch gradient
 * descent, and outputs training statistics for progress monitoring.
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
public class NeuralNetTrainer {
	private NeuralNet net;
	private Map<double[], double[]> dataMaster;
	private List<double[]> dataSampler;
	private Random random;

	private PrintStream observer;

	/**
	 * Creates a new trainer for the specified neural network, designating the given data map as the training data for
	 * the net. Registers the specified stream (or null if no observer is needed) as the observer for outputting
	 * statistics during training when requested.
	 * Map is assumed to be non-null, and map input vectors of the net's input dimension to expected output
	 * vectors of the net's output dimension. Network is assumed to non-null and untrained.
	 * DO NOT PIPE IN VALIDATION OR TESTING PARTITIONS. ONLY USE TRAINING PARTITION.
	 *
	 * @param data training data map from input vectors to expected output vectors
	 * @param net the network to train
	 */
	public NeuralNetTrainer(Map<double[], double[]> data, NeuralNet net, PrintStream observer) {
		assert data != null;
		assert net != null;
		assert data.keySet().iterator().next().length == net.getInputDim();
		assert data.values().iterator().next().length == net.getOutputDim();
		dataMaster = new HashMap<>();
		dataSampler = new ArrayList<>();
		for (double[] input : data.keySet()) {
			dataMaster.put(input, data.get(input));
		}
		refillSampler();
		this.net = net;
		this.observer = observer; // null is ok
		random = new Random(1);
	}

	/**
	 * Trains the neural network using a specified number of gradient descent steps of the specified size, drawing a
	 * randomly selected batch of the specified size each time a step is performed. Records intermittent validation
	 * statistics to the observer stream, if observing has been requested and an observer exists.
	 * Iterations and stepSize are assumed to be positive.
	 * Batch size assumed to be positive and upper bounded by the total size of the input data.
	 *
	 * @param iterations the amount of gradient descent iterations to perform
	 * @param stepSize   the scale factor for weight adjustment
	 * @param batchSize  the size of training batches to draw per gradient descent iteration
	 * @param observed   whether or not to output intermittent statistics during training
	 */
	public void train(int iterations, int stepSize, int batchSize, boolean observed) {
		int validationSize = dataMaster.size() / 100 + 1; // at least 0
		for (int i = 0; i < iterations; i++) {
			net.gradientStep(sample(batchSize), stepSize);
			if (observed && observer != null) {
				observer.printf("Loss after step %d: %.2f\n", i, validate(validationSize));
			}
		}
	}

	/**
	 * Tests and returns the net's estimated average loss using a batch of specified size of training data. Estimate not
	 * guaranteed to be within any specific degree of accuracy of the net's true average error over all samples.
	 * Batch size assumed to be positive and upper bounded by the total size of the input data.
	 *
	 * @return the net's estimated average loss
	 */
	public double validate(int batchSize) {
		assert batchSize > 0 && batchSize < dataMaster.size();
		double loss = 0;
		Iterator<double[]> keysIterator = dataMaster.keySet().iterator();
		for (int i = 0; i < batchSize; i++) {
			double[] sample = keysIterator.next();
			loss += net.calculateLoss(sample, dataMaster.get(sample));
		}
		return loss / batchSize;
	}

	/**
	 * Randomly samples a batch of training points from the total set of training data without replacement, and returns
	 * the mini batch. Once the total data set is exhausted, will re-draw from the original set. Batch size assumed to
	 * be positive and upper bounded by the total size of the input data.
	 *
	 * @param batchSize the number of data points to draw
	 * @return the generated mini batch
	 */
	private Map<double[], double[]> sample(int batchSize) {
		assert batchSize > 0 && batchSize < dataMaster.size();
		Map<double[], double[]> miniBatch = new HashMap<>();
		for (int i = 0; i < batchSize; i++) {
			if (dataSampler.isEmpty()) {
				refillSampler();
			}
			int r = random.nextInt(dataSampler.size());
			double[] sampleInput = dataSampler.get(r);
			dataSampler.remove(r);
			miniBatch.put(sampleInput, dataMaster.get(sampleInput));
		}
		return miniBatch;
	}

	/**
	 * Fills the sampler with a copy of the master. Sampler will be emptied of previous values.
	 */
	private void refillSampler() {
		dataSampler.clear();
		assert dataSampler.isEmpty();
		dataSampler.addAll(dataMaster.keySet());
	}

	/**
	 * Returns the neural network that this trainer operates on.
	 *
	 * @return the neural net
	 */
	private NeuralNet getNeuralNet() {
		return net;
	}

}
