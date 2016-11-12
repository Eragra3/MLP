using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearRegression;
using MLP.MnistHelpers;
using Newtonsoft.Json;

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
                _layers[i] = new Layer(sizes[i], sizes[i + 1], i == sizes.Length - 1);
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

            return output;
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
            var outputLayer = _layers[outputLayerIndex];
            var outputLayerWeights = outputLayer.Weights;
            var outputLayerActivation = activations[outputLayerIndex];
            var outputLayerOutput = outputs[outputLayerIndex];

            var outputLayerOutputDerivative = outputLayer.ActivationFunctionPrime(outputLayerOutput);

            var delta = (outputLayerActivation - expectedOutput).PointwiseMultiply(outputLayerOutputDerivative);

            nablaBiases[outputLayerIndex] = delta;
            nablaWeights[outputLayerIndex] = delta.OuterProduct(activations[outputLayerIndex - 1]);
            #endregion

            for (int layerIndex = _layers.Length - 2; layerIndex >= 0; layerIndex--)
            {
                var output = outputs[layerIndex];
                var outputPrime = _layers[layerIndex].ActivationFunctionPrime(output);
                var nextLayerWeights = _layers[layerIndex + 1].Weights;
                var prevLayerActivation = layerIndex == 0 ? inputs : activations[layerIndex - 1];

                delta = nextLayerWeights.Transpose().Multiply(delta).PointwiseMultiply(outputPrime);
                nablaBiases[layerIndex] = delta;
                nablaWeights[layerIndex] = delta.OuterProduct(prevLayerActivation);
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
                Console.WriteLine($"\tsizes- {JsonConvert.SerializeObject(_sizes)}");
                Console.WriteLine($"\tlearning rate - {_learningRate}");
                //Console.WriteLine($"\tmomentum- {_momentum}"););
                Console.WriteLine($"\terror threshold - {errorTreshold}");
                Console.WriteLine($"\tmax epochs - {maxEpochs}");
            }

            //Debugger.Launch();

            while (errorSum > errorTreshold && epoch < maxEpochs)
            {
                epoch++;
                errorSum = 0;

                foreach (var item in HelperFunctions.RandomPermutation(trainingSet))
                {
                    var bpResult = Backpropagate(item.Values, item.ExpectedSolution);

                    for (int i = 0; i < layersCount - 1; i++)
                    {
                        nablaBiases[i].Add(bpResult.Biases[i], nablaBiases[i]);
                        nablaWeights[i].Add(bpResult.Weights[i], nablaWeights[i]);
                    }

                    var solution = bpResult.Solution;
                    var expectedSolution = item.ExpectedSolution;

                    errorSum += solution.Map2((y, o) => Math.Pow(y - o, 2), expectedSolution).Sum();
                }

                //Console.WriteLine(nablaWeights[0].ToString(nablaWeights[0].RowCount, nablaWeights[0].ColumnCount));
                //Console.WriteLine(_layers[0].Weights.ToString());
                //Console.WriteLine("-".PadLeft(80,'*'));

                #region update parameters
                for (int i = 0; i < layersCount; i++)
                {
                    var weights = _layers[i].Weights;
                    var weightsChange = _learningRate / trainingSet.Length * nablaWeights[i];
                    weights.Subtract(weightsChange, weights);

                    var biases = _layers[i].Biases;
                    var biasesChange = _learningRate / trainingSet.Length * nablaBiases[i];
                    biases.Subtract(biasesChange, biases);
                }
                #endregion

                if (isVerbose)
                {
                    var lastEpochError = epochErrors.LastOrDefault();
                    Console.WriteLine($"Epoch - {epoch}," +
                                      $" error - {Math.Round(errorSum / trainingSet.Length, 2)}, " +
                                      $"change - {Math.Round(lastEpochError - errorSum, 2)}");
                }

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