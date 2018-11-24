/**
 * Represents a continuous, differentiable loss (error) function. One example is weighted squared difference.
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
public interface LossFunction {
	/**
	 * Computes and returns the loss for a given output component by performing an operation involving the given calculated
	 * data, and the given expected data.
	 *
	 * @param calculated the data calculated by some approximation function
	 * @param expected   the expected data
	 * @return the computed loss
	 */
	Double loss(double calculated, double expected);
}
