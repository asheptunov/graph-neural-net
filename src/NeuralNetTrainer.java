import java.io.PrintStream;
import java.util.*;

/**
 * Represents a machine learning module that trains a neural network by performing iterative stochastic batch gradient
 * descent, and outputs training statistics for progress monitoring.
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
class NeuralNetTrainer {
	private SoftmaxCrossEntropyNeuralNet net;
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
	 * @param net  the network to train
	 */
	NeuralNetTrainer(Map<double[], double[]> data, SoftmaxCrossEntropyNeuralNet net, PrintStream observer) {
		// precondition checks
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
	 * statistics to the observer stream if observing has been requested using {@code observed} and an observer exists
	 * for this trainer. todo add monitor description
	 * <p>
	 * {@code iterations} and {@code stepSize} are assumed to be positive.
	 * {@code batchSize} assumed to be positive and upper bounded by the total size of the input data.
	 *
	 * @param iterations the amount of gradient descent iterations to perform
	 * @param stepSize   the scale factor for weight adjustment
	 * @param batchSize  the size of training batches to draw per gradient descent iteration
	 * @param observed   whether or not to output intermittent statistics during training
	 * @param momentum   the momentum term
	 * @param noise      whether or not to use gradient noise
	 */
	void train(int iterations, double stepSize, int batchSize, double momentum, boolean noise, boolean observed, ProgressBar monitor) {
		// precondition checks
		assert iterations > 0 && stepSize > 0;
		assert batchSize > 0 && batchSize < dataMaster.size();

		int validationSize = dataMaster.size() / 100 + 1; // 1% of data; at least 0
		// do observer check once instead of every iteration to save time at tens of thousands of iterations
		if (observed && observer != null) {
			if (monitor != null) { // observed and progress monitored
				for (int i = 0; i < iterations; i++) {
					net.gradientStep(sample(batchSize), stepSize, momentum, noise);
					observer.printf("%d,%.2f\n", i, validate(validationSize));
					monitor.step();
				}
				monitor.finish();
			} else { // only observed
				for (int i = 0; i < iterations; i++) {
					net.gradientStep(sample(batchSize), stepSize, momentum, noise);
					observer.printf("%d,%.2f\n", i, validate(validationSize));
				}
			}
		} else if (monitor != null) { // only progress monitored
			for (int i = 0; i < iterations; i++) {
				net.gradientStep(sample(batchSize), stepSize, momentum, noise);
				monitor.step();
			}
			monitor.finish();
		} else { // unobserved and unmonitored; fastest
			for (int i = 0; i < iterations; i++) {
				net.gradientStep(sample(batchSize), stepSize, momentum, noise);
			}
		}
	}

	/**
	 * Tests and returns the net's estimated average loss using a mini batch of training size of specified size. Estimate
	 * not guaranteed to be within any specific degree of accuracy of the net's true average error over all samples.
	 * {@code batchSize} assumed to be positive and upper bounded by the total size of the trainer's training data.
	 *
	 * @param batchSize the number of data points to draw for the mini batch
	 * @return the net's estimated average loss over the batch
	 */
	private double validate(int batchSize) {
		// precondition checks
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
	 * Tests and returns the net's estimated average loss using specified input to expected (training / testing) data
	 * mappings.
	 * {@code testData} assumed to be non-null and non-empty.
	 *
	 * @param testData the data over which to validate
	 * @return the net's estimated average loss over given data
	 */
	@Deprecated
	private double validate(Map<double[], double[]> testData) {
		// precondition checks
		assert !testData.isEmpty();

		double loss = 0;
		for (double[] input : testData.keySet()) {
			loss += net.calculateLoss(input, testData.get(input));
		}
		return loss / testData.size(); // normalize
	}

	/**
	 * Randomly samples and returns a mini batch of training points from the total set of training data, without
	 * replacement. However, once the total data set is exhausted, re-draw from the original set will occur.
	 * {@code batchSize} assumed to be positive and upper bounded by the total size of the trainer's training data.
	 *
	 * @param batchSize the number of data points to draw
	 * @return the generated mini batch
	 */
	private Map<double[], double[]> sample(int batchSize) {
		// precondition checks
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
	 * Fills the sampler with a copy of the master's input (key) set. Sampler will be emptied of previous values.
	 */
	private void refillSampler() {
		dataSampler.clear();
		assert dataSampler.isEmpty();
		dataSampler.addAll(dataMaster.keySet());
	}

}
