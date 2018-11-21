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

	/**
	 * Creates a new trainer for the specified neural network, designating the given data map as the training data for
	 * the net. Map is assumed to be non-null, and map input vectors of the net's input dimension to expected output
	 * vectors of the net's output dimension. Network is assumed to non-null and untrained.
	 * DO NOT PIPE IN VALIDATION OR TESTING PARTITIONS. ONLY USE TRAINING PARTITION.
	 *
	 * @param data training data map from input vectors to expected output vectors
	 * @param net the network to train
	 */
	public NeuralNetTrainer(Map<double[], double[]> data, NeuralNet net) {
		assert data != null;
		assert net != null;
		assert data.keySet().iterator().next().length == net.getInputDim();
		assert data.values().iterator().next().length == net.getOutputDim();
		for (double[] input : data.keySet()) {
			dataMaster.put(input, data.get(input));
		}
		refillSampler();
		random = new Random(1);
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

}
