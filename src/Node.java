import java.util.*;

/**
 * Represents a single node in a directed graph with some data that can be accessed and modified. The node is
 * bidirectional, in that it stores both its children and its parents. Edges between nodes are labeled.
 *
 * @param <E> the type stored by the node
 * @param <T> the type for edge labels
 *
 * @author Andriy Sheptunov
 * @since November 2018
 */
public class Node<E, T> {
	private E data; // data in the node
	private Map<Node<E, T>, T> parents, children; // parents point to node, children are pointed to
	// using a map to type instead of collection of type restricts edge count between two nodes to 1

	/**
	 * Creates a new node with the specified data, and no parents or children. Data is assumed to be non-null.
	 *
	 * @param data the data of the node
	 */
	public Node(E data) {
		assert data != null;
		this.data = data; // encapsulation issue

		parents = new HashMap<>(); // hashing allows constant-time access
		children = new HashMap<>();
		checkRep();
	}

	/**
	 * Registers the given node as a parent of current node through the given edge. Assumes parent and edge are non-null.
	 * Subsequent calls to {@code hasParent(parent)} will return true.
	 *
	 * @param parent the parent node to attempt to register
	 * @return this node
	 */
	public Node<E, T> addParent(Node<E, T> parent, T edge) {
		if (parent == null) {
			throw new NullPointerException("Cannot add a null parent");
		}
		parents.put(parent, edge); // encapsulation issue
		checkRep();
		return this;
	}

	/**
	 * Registers the given node as a child of current node through the given edge. Assumes child and edge are non-null.
	 * Subsequent calls to (@code hasChild(child)} will return true.
	 *
	 * @param child the child node to attempt to register
	 * @return this node
	 */
	public Node<E, T> addChild(Node<E, T> child, T edge) {
		if (child == null) {
			throw new NullPointerException("Cannot add a null parent");
		}
		children.put(child, edge);
		checkRep();
		return this;
	}

	/**
	 * Returns true iff has a parent equal to the given parent node. Parent assumed to be non-null
	 *
	 * @param parent the parent to check for
	 * @return true if one of the node's parents is equal to parent, false otherwise.
	 */
	public boolean hasParent(Node<E, T> parent) {
		assert parent != null;
		return parents.containsKey(parent);
	}

	/**
	 * Returns true iff has a child equal to the given child node. Child assumed to be non-null
	 *
	 * @param child the child to check for
	 * @return true if one of the node's children is equal to the child, false otherwise.
	 */
	public boolean hasChild(Node<E, T> child) {
		assert child != null;
		return children.containsKey(child);
	}

	/**
	 * Returns the edge value between this node and the specified parent. Assumes that {@code hasParent(parent) == true}.
	 *
	 * @param parent the parent node to get the edge to
	 * @return the edge value to the parent node
	 */
	public T edgeToParent(Node<E, T> parent) {
		assert hasParent(parent);
		return parents.get(parent);
	}

	/**
	 * Returns the edge value between this node and the specified child. Assumes that {@code hasChild(child) == true}.
	 *
	 * @param child the child node to get the edge to
	 * @return the edge value to the child node
	 */
	public T edgeToChild(Node<E, T> child) {
		assert hasChild(child);
		return children.get(child);
	}

	/**
	 * Returns the data at this node.
	 *
	 * @return the node's data
	 */
	public E data() {
		return data; // big encapsulation issue
	}

	/**
	 * Sets the data at this node. Assumes data is non-null.
	 *
	 * @param data the data to attempt to store
	 */
	public void setData(E data) {
		assert data != null;
		this.data = data;
	}

	/**
	 * Returns the set of all parent nodes, or an empty set if there are no parents.
	 *
	 * @return the set of all parents
	 */
	public Set<Node<E, T>> parents() {
		return Collections.unmodifiableSet(parents.keySet()); // safe
	}

	/**
	 * Returns the set of all children nodes, or an empty set if there are no children.
	 *
	 * @return the set of all children
	 */
	public Set<Node<E, T>> children() {
		return Collections.unmodifiableSet(children.keySet());
	}

	/**
	 * Standard hashing function. Returns the hash code for this node.
	 *
	 * @return the hash code for this node
	 */
	@Override
	public int hashCode() {
		int result = data.hashCode();
		result = result * 31 + parents.hashCode();
		result = result * 31 + children.hashCode();
		return result;
	}

	/**
	 * Standard equality operation. Returns true iff this node is equal to the given object.
	 *
	 * @param obj the object to compare this node to
	 * @return true if this node and the object are equal, false otherwise
	 */
	@Override
	public boolean equals(Object obj) {
		if (!(obj instanceof Node)) {
			return false;
		}
		Node<E, T> other = (Node<E, T>) obj;
		return data.equals(other.data) && parents.equals(other.parents) && children.equals(other.children);
	}

	/**
	 * Checks the rep invariant.
	 */
	private void checkRep() {
		assert data != null;
		assert parents != null;
		assert children != null;
	}
}
