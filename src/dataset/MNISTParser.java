package dataset;

import static java.lang.String.format;
import java.io.ByteArrayOutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MNISTParser {
	public static final int LABEL_FILE_MAGIC_NUMBER = 2049;
	public static final int IMAGE_FILE_MAGIC_NUMBER = 2051;

	public static int[] getLabels(String infile) {

		ByteBuffer bb = ByteBuffer.wrap(loadFile(infile));
		assertMagicNumber(LABEL_FILE_MAGIC_NUMBER, bb.getInt());

		int numLabels = bb.getInt();
		int[] labels = new int[numLabels];

		for (int i = 0; i < numLabels; ++i)
			labels[i] = bb.get() & 0xFF; 

		return labels;
	}

	public static int[][] getDataset(String infile, boolean isBinaryValue) {

		ByteBuffer bb = ByteBuffer.wrap(loadFile(infile));
		assertMagicNumber(IMAGE_FILE_MAGIC_NUMBER, bb.getInt());

		final int sizeOfDataset = bb.getInt();
		int numberOfRows = bb.getInt();
		int numberOfColumns = bb.getInt();

		int[][] dataset = new int[sizeOfDataset][numberOfRows*numberOfColumns];

		// for each image vector to get in the dataset
		for (int rowOfDataset = 0; rowOfDataset < sizeOfDataset; rowOfDataset++) {
			dataset[rowOfDataset] = getOneInput(isBinaryValue, numberOfRows, numberOfColumns, bb);
		}
		
		return dataset;
	}

	/* This should make all the 28x28=784 elements all on the same line
	 * as input layer/visible layer for RBM, DBN input layer */ 
	private static int[] getOneInput(boolean isBinaryValue, int numberOfRows, int numberOfColumns, ByteBuffer bb) {
		ArrayList<Integer> image = new ArrayList<>(); 

		for (int i = 0; i < numberOfRows; i++) {	
			for (int col = 0; col < numberOfColumns; ++col) {
				image.add((bb.get() & 0xFF));
			}	
		}
		// In this way transform the ArrayList of Double in an array of double in the same function
		int[] unboxedArray;
		
		if(!isBinaryValue) {
			unboxedArray = unbox(image);
		} else {
			unboxedArray = unboxInBinaryValue(image);
		}
		return unboxedArray;	
	}	
	
	private static int[] unbox(ArrayList<Integer> list) {
		int[] unboxedArray = new int[list.size()];
		for(int i = 0; i < list.size(); i++) {
			unboxedArray[i] = list.get(i);
		}
		return unboxedArray;
	}
	
	private static int[] unboxInBinaryValue(ArrayList<Integer> list) {
		int[] unboxedArray = new int[list.size()];
		for(int i = 0; i < list.size(); i++) {
			int pixel = list.get(i);
			// transforming pixel in binary value
			// Mnist max = 255, min = 0
			// Rule of thumb from dl4j at https://deeplearning4j.org/rbm-mnist-tutorial.html
			if(pixel > 127) {
				pixel = 1;
			} else {
				pixel = 0;
			}
			unboxedArray[i] = pixel;
		}
		return unboxedArray;
	}


	public static List<int[][]> getImages(String infile) {
		ByteBuffer bb = ByteBuffer.wrap(loadFile(infile));

		assertMagicNumber(IMAGE_FILE_MAGIC_NUMBER, bb.getInt());

		int numImages = bb.getInt();
		//		System.out.println("numImages: " + numImages); //10000
		int numRows = bb.getInt();
		//		System.out.println("numRows: " + numRows); //28
		int numColumns = bb.getInt();
		//		System.out.println("numColumns: " + numColumns);//28
		List<int[][]> images = new ArrayList<>();

		for (int i = 0; i < numImages; i++)
			images.add(readImage(numRows, numColumns, bb));

		return images;
	}

	private static int[][] readImage(int numRows, int numCols, ByteBuffer bb) {
		int[][] image = new int[numRows][];
		for (int row = 0; row < numRows; row++)
			image[row] = readRow(numCols, bb);
		return image;
	}

	private static int[] readRow(int numCols, ByteBuffer bb) {
		int[] row = new int[numCols];
		for (int col = 0; col < numCols; ++col)
			row[col] = bb.get() & 0xFF; // To unsigned
		return row;
	}

	public static void assertMagicNumber(int expectedMagicNumber, int magicNumber) {
		if (expectedMagicNumber != magicNumber) {
			switch (expectedMagicNumber) {
			case LABEL_FILE_MAGIC_NUMBER:
				throw new RuntimeException("This is not a label file.");
			case IMAGE_FILE_MAGIC_NUMBER:
				throw new RuntimeException("This is not an image file.");
			default:
				throw new RuntimeException(
						format("Expected magic number %d, found %d", expectedMagicNumber, magicNumber));
			}
		}
	}

	public static byte[] loadFile(String infile) {
		try {
			RandomAccessFile f = new RandomAccessFile(infile, "r");
			FileChannel chan = f.getChannel();
			long fileSize = chan.size();
			ByteBuffer bb = ByteBuffer.allocate((int) fileSize);
			chan.read(bb);
			bb.flip();
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			for (int i = 0; i < fileSize; i++)
				baos.write(bb.get());
			chan.close();
			f.close();
			return baos.toByteArray();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static String renderImage(int[][] image) {
		StringBuffer sb = new StringBuffer();

		for (int row = 0; row < image.length; row++) {
			sb.append("|");
			for (int col = 0; col < image[row].length; col++) {
				int pixelVal = image[row][col];
				if (pixelVal == 0)
					sb.append(" ");
				else if (pixelVal < 256 / 3)
					sb.append(".");
				else if (pixelVal < 2 * (256 / 3))
					sb.append("x");
				else
					sb.append("X");
			}
			sb.append("|\n");
		}

		return sb.toString();
	}

	public static String repeat(String s, int n) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < n; i++)
			sb.append(s);
		return sb.toString();
	}
}