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
	private NeuralNetTrainer trainer;
	private Map<double[], double[]> trainingPartition;
	private Map<double[], double[]> testingPartition;

	/**
	 * Creates a new MNIST trainer, importing the training and testing partitions of the MNIST dataset, and starting
	 * with a blank neural net of the specified dimensions for testing.
	 * Hidden layer count is assumed to be non-negative, and hidden layer depth is assumed to be 0.
	 *
	 * @param hiddenLayerCount the number of hidden layers to use in the net
	 * @param hiddenLayerDim the dimension of hidden layers in the net
	 * @throws IOException if an I/O error has occurred
	 */
	public MNISTTrainer(int hiddenLayerCount, int hiddenLayerDim) throws IOException {
		assert hiddenLayerCount >= 0;
		assert hiddenLayerDim > 0;
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
		for (int i = 0; i < trainingLabelSamples; i++) {
			trainingPartition.put(readImage(trainingImages, trainingImageBytes), readLabel(trainingLabels));
		}

		// parse test database
		assert readInt(testLabels) == 2049;
		assert readInt(testImages) == 2051;
		int testLabelsSamples = readInt(testLabels);
		int testImagesSamples = readInt(testImages);
		assert testLabelsSamples == testImagesSamples;
		int testImageBytes = readInt(testImages) * readInt(testImages);
		assert trainingImageBytes == testImageBytes; // ensure same dim as training database
		for (int i = 0; i < testImagesSamples; i++) {
			testingPartition.put(readImage(testImages, testImageBytes), readLabel(testLabels));
		}

		// create observer
//		PrintStream observer = new PrintStream(new File("obs/observer" + System.nanoTime() % 9999 + ".txt"));

		// initialize network
		NeuralNet net = new NeuralNet(trainingImageBytes, 10, hiddenLayerDim, 2+hiddenLayerCount,
				a -> 1.0 / (1 + Math.exp(-a)), // logistic sigmoid
				a-> (1.0 / (1 + Math.exp(-a)))*(1-(1.0 / (1 + Math.exp(-a)))), // sigmoid prime
				(c, e) -> 0.5*Math.pow(e - c, 2), // weighted difference of squares loss
				(c, e) -> (c - e)); // loss prime
		trainer = new NeuralNetTrainer(trainingPartition, net, System.out);
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
		double[] output = new double[bytes];
		for (int i = 0; i < bytes; i++) {
			output[i] = in.read();
		}
		return output;
	}

	/**
	 * Reads a byte from the given input stream, and returns its value indexed as itself in an array of 10 zeroes.
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
		output[label] = label;
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
	public void train(int iterations, int stepSize, int batchSize) {
		trainer.train(iterations, stepSize, batchSize, true);
	}

	public static void main(String[] args) {
		try {
			MNISTTrainer trainer = new MNISTTrainer(4, 16);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
