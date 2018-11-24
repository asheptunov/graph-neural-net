import java.io.*;
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
	private NeuralNet net;
	private NeuralNetTrainer trainer;
	private Map<double[], double[]> trainingPartition;
	private Map<double[], Integer> testingPartition;

	private boolean observed;

	/**
	 * Creates a new MNIST trainer, importing the training and testing partitions of the MNIST dataset, and starting
	 * with a blank neural net of the specified dimensions for testing.
	 * Hidden layer count is assumed to be non-negative, and hidden layer depth is assumed to be 0.
	 *
	 * @param hiddenLayerCount the number of hidden layers to use in the net
	 * @param hiddenLayerDim the dimension of hidden layers in the net
	 * @param observed whether or not training progress should be assessed and written in real-time
	 * @throws IOException if an I/O error has occurred
	 */
	public MNISTTrainer(int hiddenLayerCount, int hiddenLayerDim, boolean observed) throws IOException {
		assert hiddenLayerCount >= 0;
		assert hiddenLayerDim > 0;
		this.observed = observed;
		// import MNIST
		FileInputStream trainingLabels = new FileInputStream(new File("data/train-labels-idx1-ubyte"));
		FileInputStream trainingImages = new FileInputStream(new File("data/train-images-idx3-ubyte"));
		FileInputStream testLabels= new FileInputStream(new File("data/t10k-labels-idx1-ubyte"));
		FileInputStream testImages = new FileInputStream(new File("data/t10k-images-idx3-ubyte"));

		// initialize partitions
		trainingPartition = new HashMap<>();
		testingPartition = new HashMap<>();

		// parse training database
		assert readInt(trainingLabels) == 2049;
		assert readInt(trainingImages) == 2051;
		int trainingLabelSamples = readInt(trainingLabels);
		int trainingImageSamples = readInt(trainingImages);
		assert trainingLabelSamples == trainingImageSamples;
		int trainingImageBytes = readInt(trainingImages) * readInt(trainingImages);
		assert trainingImageBytes == 784;
		ProgressBar pb1 = new ProgressBar(15, trainingLabelSamples);
		for (int i = 0; i < trainingLabelSamples; i++) {
			trainingPartition.put(readImage(trainingImages, trainingImageBytes), readLabel(trainingLabels));
			pb1.step();
		}
		pb1.finish();

		// parse testOnTestData database
		assert readInt(testLabels) == 2049;
		assert readInt(testImages) == 2051;
		int testLabelSamples = readInt(testLabels);
		int testImageSamples = readInt(testImages);
		assert testLabelSamples == testImageSamples;
		int testImageBytes = readInt(testImages) * readInt(testImages);
		assert trainingImageBytes == testImageBytes; // ensure same dim as training database
		ProgressBar pb2 = new ProgressBar(10, testImageSamples);
		for (int i = 0; i < testImageSamples; i++) {
			testingPartition.put(readImage(testImages, testImageBytes), testLabels.read());
			pb2.step();
		}
		pb2.finish();

		// create observer
		PrintStream observer = null;
		if (observed) {
			long obsID = System.nanoTime() % 99999;
			observer = new PrintStream(new File("obs/observer" + obsID + ".csv"));
			System.out.println("\nUsing observer " + obsID);
		} else {
			System.out.println("\nNot using observer");
		}

		// initialize network
		net = new NeuralNet(trainingImageBytes, 10, hiddenLayerDim, 2 + hiddenLayerCount,
				a -> (a > 0) ? a : 0.01 * a, // Leaky ReLU
				a -> (a <= 0.0) ? 0.01 : 1.0, // step func (Leaky ReLU derivative)
				(c, e) -> 0.5 * (e - c) * (e - c), // weighted difference of squares loss
				(c, e) -> (c - e)); // loss prime
		trainer = new NeuralNetTrainer(trainingPartition, net, observer);
	}

	/**
	 * Reads four bytes from the given input stream, converting them to a Big-Endian integer.
	 * Input stream is assumed to be non-null, and to accommodate a full 4 bytes to be read.
	 *
	 * @param in the file stream to read from
	 * @return the generated MSB-first integer
	 * @throws IOException if an I/O error occurred
	 */
	private int readInt(FileInputStream in) throws IOException {
		return (in.read() << 24) + (in.read() << 16) + (in.read() << 8) + (in.read());
	}

	/**
	 * Reads an array of the specified number of singular bytes from the given input stream. Returns an array of pixels
	 * representing an image. The first element in output will be the top left pixel, and subsequent element represent
	 * pixels going left-to-right, top-to-bottom.
	 * Input stream is assumed to be non-null, and to accommodate a full {@code bytes} number of bytes to be read.
	 * Bytes is assumed to be non-negative.
	 *
	 * @param in    the file stream to read from
	 * @param bytes the number of pixels in the image, corresponding to the number of bytes to be read
	 * @return an array of the bytes in the image
	 * @throws IOException if an I/O error occurred
	 */
	private double[] readImage(FileInputStream in, int bytes) throws IOException {
		byte[] raw = new byte[bytes];
		int read = in.read(raw);
		assert read == bytes;
		double[] output = new double[bytes];
		for (int i = 0; i < bytes; i++) {
			output[i] = (raw[i] & 0xff) / 255.0; // pull into 0-1 range
		}
		return output;
	}

	/**
	 * Reads a byte from the given input stream, and returns 1 at the index of the value in an array with the 9 other
	 * values being zeroes.
	 * Input stream is assumed to be non-null, and to accommodate a full 1 byte to be read.
	 *
	 * @param in the file stream to read from
	 * @return an array where the read byte is indexed as itself and all 9 other values are 0
	 * @throws IOException if an I/O error occurred
	 */
	private double[] readLabel(FileInputStream in) throws IOException {
		double[] output = new double[10];
		int label = in.read(); // reads 1-byte label
		assert label >= 0 && label <= 9;
		output[label] = 1.0;
		return output;
	}

	/**
	 * Trains the neural network on the MNIST training set using a specified number of gradient descent steps of the
	 * specified size, drawing a randomly selected batch of the specified size each time a step is performed.
	 *
	 * @param iterations the amount of gradient descent iterations to perform
	 * @param stepSize   the scale factor for weight adjustment
	 * @param batchSize  the size of training batches to draw per gradient descent iteration
	 */
	public void train(int iterations, double stepSize, int batchSize) {
		trainer.train(iterations, stepSize, batchSize, observed);
	}

	/**
	 * Tests the neural network on the MNIST testing set and returns the classification hit rate.
	 *
	 * @param verbose whether or not to write status to the console in real time
	 * @return the hit rate for the neural net over the testing set
	 */
	public double testOnTestData(boolean verbose) {
		double hits = 0;
		for (double[] input : testingPartition.keySet()) {
			int expected = testingPartition.get(input);
			double[] outputVector = net.propagate(input);
			int actual = 0; // index of largest output activation
			assert outputVector.length == 10;
			for (int i = 0; i < 10; i++) {
				if (outputVector[i] >= outputVector[actual]) actual = i;
			}
			if (verbose) {
				System.out.print("vector: ");
				for (int i = 0; i < outputVector.length; i++) {
					System.out.printf("%.5f ", outputVector[i]);
				}
				System.out.println("\nchoice: " + actual);
				System.out.println("expected: "+ expected);
			}
			if (actual == expected) hits++;
		}
		return hits / testingPartition.size();
	}

	/**
	 * Tests the neural network on the MNIST training set and returns the classification hit rate.
	 *
	 * @param verbose whether or not to write status to the console in real time
	 * @return the hit rate for the neural net over the training set
	 */
	public double testOnTrainingData(boolean verbose) {
		double hits = 0;
		for (double[] input : trainingPartition.keySet()) {
			double[] expectedRaw = trainingPartition.get(input);
			int expected = 0;
			for (int i = 0; i < expectedRaw.length; i++) {
				if (expectedRaw[i] == 1) expected = i;
			}
			double[] outputVector = net.propagate(input);
			int actual = 0; // index of largest output activation
			assert outputVector.length == 10;
			for (int i = 0; i < 10; i++) {
				if (outputVector[i] >= outputVector[actual]) actual = i;
			}
			if (verbose) {
				System.out.print("vector: ");
				for (double v : outputVector) {
					System.out.printf("%.5f ", v);
				}
				System.out.println("\nchoice: " + actual);
				System.out.println("expected: "+ expected);
			}
			if (actual == expected) hits++;
		}
		return hits / trainingPartition.size();
	}

	public static void main(String[] args) {
		try {
			// init and pre-test
			MNISTTrainer trainer = new MNISTTrainer(4, 16, false);
			System.out.printf("\n%1.2f%% hit rate before training.\n", trainer.testOnTestData(false) * 100.);

			// train
			System.out.println("\nTraining...");
			long time = System.nanoTime();
			trainer.train(1 << 14, .24, 8);
			System.out.printf("Trained in %d second(s).\n", (System.nanoTime() - time) / 1000000000);

			// post-test
			System.out.printf("\n%1.2f%% hit rate after training.\n", trainer.testOnTestData(false) * 100.);

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Generates and returns an array the specified dimension containing numbers from 0.0 (inclusive) to 0.1 (exclusive).
	 *
	 * @param dim the requested array dimension
	 * @return the generated array
	 */
	@Deprecated
	private static double[] genRandomInput(int dim) {
		double[] output = new double[dim];
		for (int i = 0; i < output.length; i++) {
			output[i] = Math.random();
		}
		return output;
	}
}
