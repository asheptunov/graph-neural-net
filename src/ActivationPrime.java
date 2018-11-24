/**
 * Represents a continuous derivative of an activation function
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
public interface ActivationPrime {
	/**
	 * Applies the derivative of an activation function to the given input and returns the calculated value.
	 *
	 * @param a the input
	 * @return the calculated value
	 */
	Double func(double a);
}
