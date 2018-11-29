import java.util.Map;

interface NeuralNet {

    double[] propagate(double[] input);

    double calculateLoss(double[] input, double[] expected);

    Map<Integer, double[][]> calculateWeightGradient(double[] input, double[] expected);

    void gradientStep(Map<double[], double[]> batch, double step, double momentum, boolean noise);

    int getInputDim();

    int getOutputDim();

}
