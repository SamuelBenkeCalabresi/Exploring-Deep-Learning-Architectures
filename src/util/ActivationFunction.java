package util;

public class ActivationFunction {
	
    public static double sigmoid(double x) {
        return 1. / (1. + Math.pow(Math.E, -x));
    }

    public static double dsigmoid(double x) {
        return x * (1. - x);
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double dtanh(double x) {
        return 1. - x * x;
    }

    public static double ReLU(double x) {
        if(x > 0) {
            return x;
        } else {
            return 0.;
        }
    }
    
    public static double dReLU(double x) {
        if(x > 0) {
            return 1.;
        } else {
            return 0.;
        }
    }
}
