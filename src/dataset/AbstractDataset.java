package dataset;
/** 
 * The aim is to create a better package 
 * and handler methods from the data 
 * to be used from the machine learning models*/
public abstract class AbstractDataset {
	
	/* The data on any row of the matrix */
	protected final int[] data;
	
	public AbstractDataset(int[] data) {
		this.data = data;
	}
}
