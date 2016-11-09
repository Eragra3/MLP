using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MLP.MnistHelpers;
using Newtonsoft.Json;
using static MLP.HelperFunctions;

namespace MLP
{
    public class Mlp
    {
        [JsonProperty]
        private Layer[] _layers;

        [JsonProperty]
        private int[] _sizes;

        [JsonProperty]
        private double _learningRate;

        [JsonProperty]
        private double _momentum;

        [JsonProperty]
        private double _errorThreshold;

        /// <summary>
        /// Should only be used to create deserialized object
        /// </summary>
        public Mlp()
        {

        }

        public Mlp(double learningRate, double momentum, double errorThreshold, params int[] sizes)
        {
            _learningRate = learningRate;
            _momentum = momentum;
            _errorThreshold = errorThreshold;
            _sizes = sizes;
            _layers = new Layer[sizes.Length - 1];

            for (int i = 0; i < sizes.Length - 1; i++)
            {
                _layers[i] = new Layer(sizes[i], sizes[i + 1]);
            }
        }

        /// <summary>
        /// acts as output layer
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public Vector<double> Feedforward(Vector<double> inputs)
        {
            var output = inputs;

            for (int i = 0; i < _layers.Length; i++)
            {
                output = _layers[i].Feedforward(output);
            }

            return Sigmoid(output);
        }

        public int Compute(Vector<double> inputs)
        {
            var output = Feedforward(inputs);

            var max = output[0];
            var maxIndex = 0;

            for (int i = 1; i < output.Count; i++)
            {
                if (output[i] > max)
                {
                    max = output[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        public BackpropagationResult Backpropagate(Vector<double> inputs, Vector<double> expectedOutput)
        {
            var layersCount = _layers.Length;
            var nablaWeights = new Matrix<double>[layersCount];
            var nablaBiases = new Vector<double>[layersCount];

            var outputs = new Vector<double>[layersCount];
            var activations = new Vector<double>[layersCount];
            #region get outputs and activations
            var prevActivation = inputs;
            for (int i = 0; i < _layers.Length; i++)
            {
                var output = _layers[i].GetOutput(prevActivation);
                outputs[i] = output;
                var activation = _layers[i].GetActivation(output);
                activations[i] = activation;
                prevActivation = activation;
            }
            #endregion

            #region get output layer nablas
            var outputLayerIndex = layersCount - 1;
            var outputLayerWeights = _layers[outputLayerIndex].Weights;
            var outputLayerActivation = activations[outputLayerIndex];
            var outputLayerOutputDerivative = SigmoidPrime(outputLayerWeights.Multiply(outputs[outputLayerIndex - 1]));

            var delta = expectedOutput.Map2((y, a) => y - a, outputLayerActivation)
                            .PointwiseMultiply(outputLayerOutputDerivative);

            nablaBiases[outputLayerIndex] = delta;
            nablaWeights[outputLayerIndex] = delta.OuterProduct(activations[outputLayerIndex]);
            #endregion

            for (int layerIndex = _layers.Length - 2; layerIndex >= 0; layerIndex--)
            {
                var prevLayerOutput = layerIndex == 0 ? inputs : outputs[layerIndex - 1];
                var nextLayerWeights = _layers[layerIndex + 1].Weights;
                var weights = _layers[layerIndex].Weights;

                var sigmoidPrime = SigmoidPrime(weights.Multiply(prevLayerOutput));
                delta = nextLayerWeights.Transpose().Multiply(delta).PointwiseMultiply(sigmoidPrime);
                nablaBiases[layerIndex] = delta;
                nablaWeights[layerIndex] = delta.OuterProduct(prevLayerOutput);
            }

            var result = new BackpropagationResult
            {
                Biases = nablaBiases,
                Weights = nablaWeights,
                Input = inputs,
                Solution = activations.Last()
            };

            return result;
        }


        public TrainingResult Train(TrainingModel trainingModel)
        {
            var trainingSet = trainingModel.TrainingSet;
            var validationSet = trainingModel.ValidationSet;
            var errorTreshold = trainingModel.ErrorThreshold;
            var maxEpochs = trainingModel.MaxEpochs;
            var isVerbose = trainingModel.IsVerbose;

            IList<double> epochErrors = new List<double>();

            double errorSum = double.PositiveInfinity;
            int epoch = 0;

            var layersCount = _layers.Length;
            var lastLayerIndex = _layers.Length - 1;
            #region create nablas arrays
            var nablaWeights = new Matrix<double>[layersCount];
            var nablaBiases = new Vector<double>[layersCount];
            for (int i = 0; i < layersCount; i++)
            {
                var nextLayer = _layers[i];
                nablaBiases[i] = nextLayer.GetNewBiasesVector(true);
                nablaWeights[i] = nextLayer.GetNewWeightsMatrix(true);
            }
            #endregion

            if (isVerbose)
            {
                Console.WriteLine("Starting with params:");
                Console.WriteLine($"\tsizes- {_sizes}");
                Console.WriteLine($"\tlearning rate - {_learningRate}");
                //Console.WriteLine($"\tmomentum- {_momentum}"););
                Console.WriteLine($"\terror threshold - {errorTreshold}");
                Console.WriteLine($"\tmax epochs - {maxEpochs}");
            }

            while (errorSum > errorTreshold && epoch < maxEpochs)
            {
                epoch++;
                errorSum = 0;

                foreach (var item in trainingSet)
                {
                    var bpResult = Backpropagate(item.Values, item.ExpectedSolution);

                    for (int i = 0; i < layersCount - 1; i++)
                    {
                        nablaBiases[i] += bpResult.Biases[i];
                        nablaWeights[i] += bpResult.Weights[i];
                    }

                    var solution = bpResult.Solution;
                    var expectedSolution = item.ExpectedSolution;

                    errorSum += solution.Map2((y, o) => Math.Pow(y - o, 2), expectedSolution).Sum();
                }

                if (isVerbose) Console.WriteLine($"Epoch - {epoch}, error - {Math.Round(errorSum, 2)}");

                #region set nablas to zeroes
                for (int i = 0; i < layersCount; i++)
                {
                    nablaBiases[i].Clear();
                    nablaWeights[i].Clear();
                }
                #endregion

                epochErrors.Add(errorSum);
            }

            var trainingResult = new TrainingResult
            {
                Mlp = this,
                Epochs = epoch,
                EpochErrors = epochErrors.ToArray()
            };

            return trainingResult;
        }

        public string ToJson()
        {
            return JsonConvert.SerializeObject(this, Formatting.Indented);
        }

        public static Mlp FromJson(string json)
        {
            return JsonConvert.DeserializeObject<Mlp>(json);
        }
    }
}