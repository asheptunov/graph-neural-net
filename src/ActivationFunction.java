/**
 * Represents a continuous, differentiable activation function. One example is a logistic curve.
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
public interface ActivationFunction {
	/**
	 * Applies an activation function to the given input and returns the calculated value.
	 *
	 * @param a the input
	 * @return the calculated value
	 */
	Double func(double a);
}
