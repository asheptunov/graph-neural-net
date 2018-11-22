import java.util.Map;

/**
 * TODO
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
public class MNISTTrainer {
	// delegate instead of subclassing
	private NeuralNetTrainer trainer;
	private Map<double[], double[]> trainingPartition;
	private Map<double[], double[]> testingPartition;

	/**
	 * Creates a new MNIST trainer, importing and partitioning the MNIST dataset into a training set of the specified
	 * size, and a reserved remainder of data for validation or testing.
	 * Training fraction is assumed to be between 0.0 (exclusive), and 1.0 (inclusive).
	 *
	 * @param trainingFraction the fraction of MNIST that should be used for training.
	 */
	public MNISTTrainer(double trainingFraction) {
		assert trainingFraction > 0.0 && trainingFraction <= 1.0;
		// import MNIST

		// initialize partitions

		// partition MNIST

		// create observer

		// initialize network
	}

	/**
	 * Trains the neural network using a specified number of gradient descent steps of the specified size, drawing a
	 * randomly selected batch of the specified size each time a step is performed. Records intermittent validation
	 * statistics to the observer stream, if observing has been requested and an observer exists.
	 *
	 * @param iterations the amount of gradient descent iterations to perform
	 * @param stepSize   the scale factor for weight adjustment
	 * @param batchSize  the size of training batches to draw per gradient descent iteration
	 * @param observed   whether or not to output intermittent statistics during training
	 */
	public void train(int iterations, int stepSize, int batchSize) {
		trainer.train(iterations, stepSize, batchSize, true);
	}
}
