import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * A testing client capable of training any neural network on the MNIST dataset, and evaluating its performance.
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
class MNISTTrainer {
    private SoftmaxCrossEntropyNeuralNet net;
    private NeuralNetTrainer trainer; // trainer delegate
    private Map<double[], double[]> trainingPartition; // optimized for training (output is distribution)
    private Map<double[], Integer> testingPartition; // optimized for testing (output is classification)
    private PrintStream trainingLog;

    /**
     * Creates a new MNIST trainer, importing the training and testing partitions of the MNIST dataset, and starting
     * with a blank neural net of the specified hidden layer dimension vector.
     * Hidden layer count vector is assumed to be non-null, non-empty and contain non-negative values.
     * // todo remove hiddenlayerdim input, mnisttrainer should be net-oblivious; refactor to net interface and just take a net
     * @throws IOException if an I/O error has occurred
     */
    private MNISTTrainer(int[] hiddenLayerDims) throws IOException {
        // import MNIST
        FileInputStream trainingLabels = new FileInputStream(new File("data/train-labels-idx1-ubyte"));
        FileInputStream trainingImages = new FileInputStream(new File("data/train-images-idx3-ubyte"));
        FileInputStream testLabels = new FileInputStream(new File("data/t10k-labels-idx1-ubyte"));
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
        for (int i = 0; i < trainingLabelSamples; i++) {
            trainingPartition.put(readImage(trainingImages, trainingImageBytes), readLabel(trainingLabels));
        }

        // parse testOnTestData database
        assert readInt(testLabels) == 2049;
        assert readInt(testImages) == 2051;
        int testLabelSamples = readInt(testLabels);
        int testImageSamples = readInt(testImages);
        assert testLabelSamples == testImageSamples;
        int testImageBytes = readInt(testImages) * readInt(testImages);
        assert trainingImageBytes == testImageBytes; // ensure same dim as training database
        for (int i = 0; i < testImageSamples; i++) {
            testingPartition.put(readImage(testImages, testImageBytes), testLabels.read());
        }

        trainingLog = new PrintStream(new File("logs/trainLog.csv"));

        // initialize network // todo this should not happen here lol
        net = new SoftmaxCrossEntropyNeuralNet(trainingImageBytes, 10, hiddenLayerDims,
                a -> (a > 0) ? a : 0.01 * a, // leaky ReLU
                a -> (a <= 0.0) ? 0.01 : 1.0); // step func (derivative of differentiability-adjusted leaky ReLU)
        trainer = new NeuralNetTrainer(trainingPartition, net);
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
     * Reads an array of the specified number of sequential bytes from the given input stream. Returns an array of pixels
     * representing an image. The first element in output will be the top left pixel, and subsequent element represent
     * pixels going left-to-right, top-to-bottom.
     * Input stream is assumed to be non-null, and to accommodate a full {@code bytes} number of bytes to be read.
     * Requested byte count assumed to be non-negative.
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
     * values being zeroes, representing a target distribution for the decimal digit represented by the byte.
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
     * Trains the neural network on the MNIST training set using mini-batch gradient descent. Runs the specified number
     * of iterations of gradient descent, randomly drawing batches of the specified size, and adjusting weights proportional
     * to the given step size. Weight updates will include a the specified fraction of the previous iteration's update,
     * which can be set to 0 if no momentum is desired. If specified, gaussian noise will be applied to weight updates
     * as a function of the input parameters. Amount of noise will scale over iterations, and change with step size.
     * If specified, registers training progress with a progress bar, which will print to console. This will not affect
     * performance. If specified, writes intermittent training progress to a file, which will significantly impact training
     * time.
     *
     * @param iterations the amount of gradient descent iterations to perform
     * @param stepSize   the scale factor for weight adjustment
     * @param batchSize  the size of training batches to draw per gradient descent iteration
     * @param momentum   the momentum constant to apply per gradient descent iteration
     * @param noise      whether or not to apply gaussian gradient noise during weight adjustments
     * @param monitoring whether or not training progress should be monitored over time
     * @param logging    whether or not net performance should be assessed and logged in real-time during training
     * @throws IOException if logging was requested, and an I/O error occurred.
     */
    private void train(int iterations, double stepSize, int batchSize, double momentum, boolean noise, boolean monitoring, boolean logging) throws FileNotFoundException {
        ProgressBar pb = null;
        if (monitoring) {
            pb = new ProgressBar(10, iterations, progress -> "▯");
        }
        PrintStream log = null;
        if (logging) {
            long id = System.nanoTime() % 99999;
            log = new PrintStream(new File("logs/log" + id + ".csv"));
            System.out.println("Logging net performance to " + id);
        }
        trainer.train(iterations, stepSize, batchSize, momentum, noise, pb, log);
    }

    /**
     * Tests the neural network on the MNIST testing set and returns the classification hit rate as a decimal value
     * from 0.0 (inclusive) to 1.0 (inclusive).
     *
     * @return the hit rate for the neural net over the testing set
     */
    private double testOnTestData() {
        double hits = 0;
        for (double[] input : testingPartition.keySet()) {
            int expected = testingPartition.get(input);
            double[] outputVector = net.propagate(input);
            int actual = 0; // index of largest output activation
            assert outputVector.length == 10;
            for (int i = 0; i < 10; i++) {
                if (outputVector[i] >= outputVector[actual]) actual = i;
            }
            if (actual == expected) hits++;
        }
        return hits / testingPartition.size();
    }

    /**
     * Tests the neural network on the MNIST training set and returns the classification hit rate as a decimal value
     * from 0.0 (inclusive) to 1.0 (inclusive).
     *
     * @return the hit rate for the neural net over the training set
     */
    private double testOnTrainingData() {
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
            if (actual == expected) hits++;
        }
        return hits / trainingPartition.size();
    }

    /**
     * Logs the result of testing a neural net defined by specified parameters to a file.
     *
     * @param hluDim      the hidden layer dimensions of the neural net
     * @param iterations  the number of iterations in training
     * @param stepSize    the step size in training
     * @param batchSize   the mini-batch size in training
     * @param momentum    the momentum in training
     * @param noise       whether or not noise was used in training
     * @param trainingAcc the training set hit accuracy after training
     * @param testAcc     the test set hit accuracy after training
     */
    private void logTest(int[] hluDim, int iterations, double stepSize, int batchSize, double momentum, boolean noise, double trainingAcc, double testAcc) {
        trainingLog.print("784");
        for (int i = 0; i < hluDim.length; i++) {
            trainingLog.print("-" + hluDim[i]);
        }
        trainingLog.print("-10,");
        trainingLog.printf("%d,%.5f,%d,%.3f,", iterations, stepSize, batchSize, momentum);
        trainingLog.print(noise + ",");
        trainingLog.printf("%.5f,%.5f\n", trainingAcc, testAcc);
    }

    public static void main(String[] args) {
        try {
            // 94.5% with 784-100-50-10, 100k iterations, 0.0042 step, 2 bs, 0.9 momentum, no noise; 2 mins
            // 95% with 784-100-50-10, 100k iterations, 0.0075 step, 4 bs, 0.9 momentum, no noise; 4 mins
            // 96% with 784-100-50-10, 200k iterations, 0.0075 step, 4 bs, 0.9 momentum, no noise; 7.3 mins
            // 95% with 784-100-50-10, 100k iterations, 0.0100 step, 8 bs, 0.9 momentum, no noise; 7.2 mins
            // 95% with 784-100-50-10, 50k iterations, 0.0125 step, 16 bs, 0.9 momentum, no noise; 6.7 mins
            // 96.13% with 784-100-50-10, 200k iterations, 0.0125 step, 16 bs, 0.9 momentum, no noise; 2 hours 30 mins
            // 96% with 784-100-50-10, 200k iterations, 0.0125 step, 32 bs, 0.9 momentum, no noise; 2 hours 45 mins

            // net dim
            int[] hluDim = new int[]{300, 100};
            // hyper-parameters
            int iterations = 100000;
            double stepSize = 0.01;
            int batchSize = 16;
            double momentum = 0.9;
            boolean noise = false;
            // init
            MNISTTrainer trainer = new MNISTTrainer(hluDim);

            // train
            trainer.train(iterations, stepSize, batchSize, momentum, noise, true, false);
            // log
            trainer.logTest(hluDim, iterations, stepSize, batchSize, momentum, noise, trainer.testOnTrainingData(), trainer.testOnTestData());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
