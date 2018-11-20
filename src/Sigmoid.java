/**
 * Represents a continuous, differentiable sigmoid function. One example is a logistic curve.
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
public interface Sigmoid {
	/**
	 * Applies a sigmoid function to the given input and returns the calculated value.
	 *
	 * @param a the input
	 * @return the calculated value
	 */
	public Double func(double a);
}
