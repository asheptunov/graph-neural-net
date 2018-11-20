/**
 * Represents a continuous derivative of a sigmoid function
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
public interface SigmoidPrime {
	/**
	 * Applies the derivative of a sigmoid function to the given input and returns the calculated value.
	 *
	 * @param a the input
	 * @return the calculated value
	 */
	public Double func(double a);
}
