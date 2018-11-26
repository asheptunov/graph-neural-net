/**
 * Represents a customizable command-line progress bar that can iterate progress over time and reset itself.
 *
 * @author Andriy Sheptunov
 * @since September 2018
 */
class ProgressBar {
	// max number of graphical ticks
	private int maxTicks;
	// max number of logical steps
	private int maxSteps;
	// number of logical steps per tick
	private int stepsPerTick;

	// current tick number
	private int tick = 0;
	// current step number
	private int step = 0;

	private Ticker ticker;

	// starting time
	private long startTime;

	/**
	 * Represents a renderer of a single tick in the progress bar.
	 */
	interface Ticker {
		/**
		 * Returns a single tick of a progress bar given the current progress percentage.
		 *
		 * @param progress the progress percentage
		 * @return the rendered tick
		 */
		String render(double progress);
	}

	/**
	 * Creates a new progress bar with the given attributes
	 *
	 * @param ticks        the number of progress bar ticks displayed to the user
	 * @param steps        the total number of logical steps in the process, >= ticks
	 * @param tick         the rendering style for a given single tick
	 */
	ProgressBar(int ticks, int steps, Ticker tick) {
		assert tick != null;
		this.maxTicks = ticks;
		this.maxSteps = steps;
		this.ticker = tick;
		stepsPerTick = steps / ticks;
		start();
	}

	/**
	 * Ticks a single time and adjusts step progress accordingly
	 */
	void tick() {
		step(stepsPerTick);
	}

	/**
	 * Ticks a specified number of times and adjusts step progress accordingly
	 *
	 * @param ticks number of ticks to step by
	 */
	private void tick(int ticks) {
		step(ticks * stepsPerTick);
	}

	/**
	 * Steps a single time and displays a tick if appropriate
	 */
	void step() {
		step(1);
	}

	/**
	 * Steps a specified number of times and displays ticks if appropriate.
	 *
	 * @param steps number of steps to step by, expected to be 0 or a positive integer
	 */
	private void step(int steps) {
		while (steps != 0 && step < maxSteps) { // stop when run out of steps or hit ceil
			if (step % stepsPerTick == 0) displayTick();
			step++;
			steps--;
		}
		if (step == maxSteps) finish();
		checkRep();
	}

	/**
	 * Force completes the progress bar and resets the bar, without restarting it.
	 */
	void reset() {
		forceComplete();
		tick = 0;
		step = 0;
		finished = false;
		checkRep();
	}

	/**
	 * Steps the remaining possible number of times and displays remaining ticks
	 */
	void forceComplete() {
		while (step < maxSteps) { // step up to max
			step();
		}
		checkRep();
	}

	/**
	 * Prints a starting message and note down the starting time. Time is noted last, so printing isn't included in
	 * the lifespan.
	 */
	private void start() {
		System.out.println("Starting...");
		startTime = System.nanoTime();
	}

	/**
	 * Prints a single tick on the progress bar.
	 */
	private void displayTick() {
		System.out.print(ticker.render(step * 100.0 / maxSteps));
	}

	// whether or not steps have been completed
	private boolean finished = false;

	/**
	 * Prints a finishing message, and notes down the ending time. Note that this alone does not stop the timer or
	 * reset the progress bar in any way.
	 */
	void finish() {
		if (!finished) {
			finished = true;
			System.out.print(ticker.render(100.0));
			System.out.printf("\nFinished in %.2f seconds\n", (System.nanoTime() - startTime) / 1000000000.0);
		}
	}

	private void checkRep() {
		assert tick >= 0;
		assert tick <= maxTicks;
		assert step >= 0;
		assert step <= maxSteps;
	}

}
