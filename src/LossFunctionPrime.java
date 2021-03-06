/**
 * Represents a continuous derivative of a loss (error) function.
 */
public interface LossFunctionPrime {
	/**
	 * Computes and returns the first derivative of the loss for a given output component by performing an operation involving
	 * the given calculated data, and the given expected data.
	 *
	 * @param calculated the data calculated by some approximation function
	 * @param expected   the expected data
	 * @return the computed loss derivative
	 */
	Double func(double calculated, double expected);
}
