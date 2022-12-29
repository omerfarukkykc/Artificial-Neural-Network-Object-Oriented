import java.util.ArrayList;
import java.util.List;
import java.util.Random;

class NeuralNetworkMain {
    public static void main(String[] args) {
        int[] layers = { 2, 2, 1 };
        int epochs = 1000;
        double learningRate = 0.4;
        double[][] inputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        double[][] outputs = { { 0 }, { 1 }, { 1 }, { 0 } };
        ANN ann = new ANN(layers, ANN.ActivationFunction.ReLu, epochs, learningRate, inputs, outputs);
        ann.train();
        ann.printNetwork();
    }
}

class ANN {
    static Random random = new Random();
    static ActivationFunction activationFunction;
    private Layer inputLayer;
    private Layer outputLayer;
    private int epochs;
    private double learningRate;
    private double[][] inputs;
    private double[][] outputs;

    public ANN(int[] layers, ActivationFunction activationFunction, int epochs, double learningRate, double[][] inputs,
            double[][] outputs) {
        ANN.activationFunction = activationFunction;
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.inputs = inputs;
        this.outputs = outputs;

        inputLayer = new Layer(layers[0]);
        for (Neuron neuron : inputLayer.neurons) {
            neuron.bias = 0;
        }
        Layer previusLayer = inputLayer;
        Layer tmpLayer = null;
        for (int i = 1; i < layers.length; i++) {
            tmpLayer = new Layer(layers[i]);
            previusLayer.setNextLayer(tmpLayer);
            tmpLayer.setPreviousLayer(previusLayer);
            previusLayer = tmpLayer;
        }
        outputLayer = tmpLayer;

    }

    public void printNetwork() {
        Layer tmpLayer = inputLayer;
        while (tmpLayer != null) {
            for (Neuron neuron : tmpLayer.neurons) {
                System.out.print(" --" + neuron.value + "--");
            }
            System.out.println();
            tmpLayer = tmpLayer.nextLayer;
        }
    }

    public void calculateOutput() {
        Layer tmpLayer = inputLayer;
        while (tmpLayer != null) {
            for (Neuron neuron : tmpLayer.neurons) {
                neuron.calculateValue();
            }
            tmpLayer = tmpLayer.nextLayer;
        }
        System.out.println("Output: " + outputLayer.neurons.get(0).value);
    }

    public void backPropogation() {
        Layer tmpLayer = outputLayer;
        while (tmpLayer != null) {
            for (Neuron neuron : tmpLayer.neurons) {
                neuron.calculateError();
            }
            tmpLayer = tmpLayer.previousLayer;
        }
    }

    public void train() {
        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch: " + i);
            for (int j = 0; j < inputs.length; j++) {
                for (int k = 0; k < inputs[j].length; k++) {
                    inputLayer.neurons.get(k).value = inputs[j][k];
                }
                for (int k = 0; k < outputs[j].length; k++) {
                    outputLayer.neurons.get(k).output = outputs[j][k];
                }
                calculateOutput();
                backPropogation();
            }
        }
    }

    class Layer {
        private Layer previousLayer;
        private Layer nextLayer;
        private List<Neuron> neurons;

        public Layer(int size) {
            neurons = new ArrayList<Neuron>();
            for (int i = 0; i < size; i++) {
                neurons.add(new Neuron());
            }
            if (previousLayer == null) {
                return;
            }
        }

        public void setPreviousLayer(Layer previousLayer) {
            this.previousLayer = previousLayer;
            for (Neuron neuron : neurons) {
                for (Neuron previousNeuron : previousLayer.neurons) {
                    neuron.Inputs.add(previousNeuron);
                    neuron.InputsWeights.add(random.nextDouble() % 1);
                }
            }
        }

        public void setNextLayer(Layer nextLayer) {
            this.nextLayer = nextLayer;
            for (Neuron neuron : neurons) {
                for (Neuron nextNeuron : nextLayer.neurons) {
                    neuron.Outputs.add(nextNeuron);
                    neuron.OutputsWeights.add(random.nextDouble() % 1);
                }
            }
        }
    }

    class Neuron {
        List<Neuron> Inputs;
        List<Double> InputsWeights;
        List<Neuron> Outputs;
        List<Double> OutputsWeights;
        double value;
        double output = Double.MAX_VALUE;
        double error;
        double bias;

        public Neuron() {
            this.bias = random.nextDouble() % 1;
            this.Inputs = new ArrayList<Neuron>();
            this.InputsWeights = new ArrayList<Double>();
            this.Outputs = new ArrayList<Neuron>();
            this.OutputsWeights = new ArrayList<Double>();
        }

        void calculateValue() {
            double sum = 0;
            for (int i = 0; i < Inputs.size(); i++) {
                sum += Inputs.get(i).value * InputsWeights.get(i);
            }
            switch (ANN.activationFunction) {
                case Sigmoid:
                    value = sigmoid(sum + bias);
                    break;
                case ReLu:
                    value = ReLu(sum + bias);
                    break;
                default:
                    value = ReLu(sum + bias);
                    break;
            }
        }

        void calculateError() {
            if (output == Double.MAX_VALUE) {
                this.error = this.value * (1 - this.value);
            } else {
                this.error = this.value * (1 - this.value) * (output - this.value);
            }
            for (int i = 0; i < Outputs.size(); i++) {
                this.error *= Outputs.get(i).error * OutputsWeights.get(i);
            }
            for (int i = 0; i < Inputs.size(); i++) {
                InputsWeights.set(i, (InputsWeights.get(i) * learningRate * Inputs.get(i).value * error));
            }
        }

        double sigmoid(double x) {
            return 1 / (1 + Math.exp(-x));
        }

        double ReLu(double x) {
            return Math.max(0, x);
        }
    }

    enum ActivationFunction {
        Sigmoid,
        ReLu
    }
}