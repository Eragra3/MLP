using System;
using System.Collections.Generic;
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
            for (int i = 0; i < _layers.Length; i++)
            {
                var layerWeights = _layers[i].Weights;
                nablaBiases[i] = new DenseVector(layerWeights.RowCount);
            }

            var outputs = new Vector<double>[layersCount];
            var activations = new Vector<double>[layersCount];
            for (int i = 0; i < _layers.Length; i++)
            {
                var output = _layers[i].GetOutput(inputs);
                outputs[i] = output;
                activations[i] = _layers[i].GetActivation(output);
            }

            var outputLayerIndex = layersCount - 1;
            var outputLayerOutput = outputs[outputLayerIndex];
            var outputLayerCostDerivative = SigmoidPrime(outputLayerOutput);

            var delta = outputLayerOutput
                    .Map2((x, y) => x - y, expectedOutput)
                    .PointwiseMultiply(outputLayerCostDerivative);

            nablaBiases[outputLayerIndex] = delta;
            nablaWeights[outputLayerIndex] = delta.OuterProduct(activations[outputLayerIndex]);

            for (int layerIndex = _layers.Length - 2; layerIndex >= 0; layerIndex--)
            {
                var sigmoidPrime = Sigmoid(outputs[layerIndex]);
                delta = _layers[layerIndex].Weights.Transpose().Multiply(delta).PointwiseMultiply(sigmoidPrime);
                nablaBiases[layerIndex] = delta;
                nablaWeights[layerIndex] = delta.OuterProduct(outputs[layerIndex + 1]);
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

            IList<double> epochErrors = new List<double>();

            double errorSum = double.PositiveInfinity;
            int epoch = 0;

            var layersCount = _layers.Length;
            #region create nablas arrays
            var nablaWeights = new Matrix<double>[layersCount];
            var nablaBiases = new Vector<double>[layersCount];
            for (int i = 0; i < layersCount; i++)
            {
                var layer = _layers[i];
                nablaBiases[i] = layer.GetNewBiasesVector(true);
                nablaWeights[i] = layer.GetNewWeightsMatrix(true);
            }
            #endregion

            while (errorSum > errorTreshold && epoch < maxEpochs)
            {
                epoch++;
                errorSum = 0;

                foreach (var item in trainingSet)
                {
                    var bpResult = Backpropagate(item.Values, item.ExpectedSolution);

                    for (int i = 0; i < layersCount; i++)
                    {
                        nablaBiases[i].Map2((nb1, nb2) => nb1 + nb2, bpResult.Biases[i]);
                        nablaWeights[i].Map2((w1, w2) => w1 + w2, bpResult.Weights[i]);
                    }

                    var solution = bpResult.Solution;
                    var expectedSolution = item.ExpectedSolution;

                    errorSum += solution.Map2((y, o) => Math.Pow(y - o, 2), expectedSolution).Sum();
                }

                #region set nablas to zeroes
                for (int i = 0; i < layersCount; i++)
                {
                    var layer = _layers[i];
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