import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.HashMap;
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
	 * Creates a new MNIST trainer, importing the training and testing partitions of the MNIST dataset, and starting
	 * with a blank neural net of the specified dimensions for testing.
	 * Hidden layer count is assumed to be non-negative, and hidden layer depth is assumed to be 0.
	 *
	 * @param hiddenLayers the number of hidden layers to use in the net
	 * @param hiddenLayerDepth the dimension of hidden layers in the net
	 */
	public MNISTTrainer(int hiddenLayers, int hiddenLayerDepth) throws FileNotFoundException {
		assert hiddenLayers >= 0;
		assert hiddenLayerDepth > 0;
		// import MNIST
		FileInputStream mnistTrainingImages = new FileInputStream(new File("data/mnistTrainingImages.idk"));
		FileInputStream mnistTrainingLabels = new FileInputStream(new File("data/mnistTrainingLabels.idk"));
		FileInputStream mnistTestImages = new FileInputStream(new File("data/mnistTestImages.idk"));
		FileInputStream mnistTestLabels= new FileInputStream(new File("data/mnistTestLabesl.idk"));

		// initialize partitions
		trainingPartition = new HashMap<>();
		testingPartition = new HashMap<>();

		// parse MNIST


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
