public interface LossFunction {
	/**
	 * Computes and returns the loss for a given output component by performing an operation involving the given calculated
	 * data, and the given expected data.
	 *
	 * @param calculated the data calculated by some approximation function
	 * @param expected   the expected data
	 * @return the computed loss
	 */
	public Double loss(Double calculated, Double expected);
}
