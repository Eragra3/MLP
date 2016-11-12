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

            return output.MaximumIndex();
        }

        public BackpropagationResult Backpropagate(Vector<double> inputs, Vector<double> expectedOutput)
        {
            var layersCount = _layers.Length;
            var nablaWeights = new Matrix<double>[layersCount];
            var nablaBiases = new Vector<double>[layersCount];

            var nets = new Vector<double>[layersCount];
            var activations = new Vector<double>[layersCount + 1];
            activations[0] = inputs;
            #region get nets and activations
            var prevActivation = inputs;
            for (int i = 0; i < _layers.Length; i++)
            {
                var net = _layers[i].GetNet(prevActivation);
                nets[i] = net;
                var activation = _layers[i].GetActivation(net);
                activations[i + 1] = activation;
                prevActivation = activation;
            }
            #endregion

            #region get output layer nablas
            var outputLayerIndex = layersCount - 1;
            var outputLayer = _layers[outputLayerIndex];
            var outputLayerWeights = outputLayer.Weights;
            var outputLayerActivation = activations.Last();
            var outputLayerNet = nets[outputLayerIndex];

            var outputLayerNetDerivative = outputLayer.ActivationFunctionPrime(outputLayerNet);

            var delta = (outputLayerActivation - expectedOutput).PointwiseMultiply(outputLayerNetDerivative);

            nablaBiases[outputLayerIndex] = delta;
            nablaWeights[outputLayerIndex] = delta.OuterProduct(activations[outputLayerIndex]);
            #endregion

            for (int layerIndex = _layers.Length - 2; layerIndex >= 0; layerIndex--)
            {
                var layer = _layers[layerIndex];
                var input = activations[layerIndex];
                var net = nets[layerIndex];
                var netPrime = layer.ActivationFunctionPrime(net);
                var nextLayerWeights = _layers[layerIndex + 1].Weights;

                delta = nextLayerWeights.Transpose().Multiply(delta).PointwiseMultiply(netPrime);
                nablaBiases[layerIndex] = delta;
                nablaWeights[layerIndex] = delta.OuterProduct(input);
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
            var batchSize = trainingModel.BathSize;

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

                foreach (var item in HelperFunctions.RandomPermutation(trainingSet).Take(batchSize))
                {
                    var bpResult = Backpropagate(item.Values, item.ExpectedSolution);

                    for (int i = 0; i < layersCount - 1; i++)
                    {
                        nablaBiases[i] = bpResult.Biases[i] + nablaBiases[i];
                        nablaWeights[i] = bpResult.Weights[i] + nablaWeights[i];
                    }

                    var solution = bpResult.Solution;
                    var expectedSolution = item.ExpectedSolution;

                    errorSum += solution.Map2((y, o) => Math.Pow(y - o, 2), expectedSolution).Sum();
                }

                //Console.WriteLine(nablaWeights[0].ToString(nablaWeights[0].RowCount, nablaWeights[0].ColumnCount));
                //Console.WriteLine(_layers[0].Weights.ToString());
                //Console.WriteLine("-".PadLeft(80,'*'));
                Console.WriteLine(MlpTrainer.Evaluate(this, validationSet).Percentage);

                #region update parameters
                for (int i = 0; i < layersCount; i++)
                {
                    var weights = _layers[i].Weights;
                    var weightsChange = _learningRate / batchSize * nablaWeights[i];
                    _layers[i].Weights = weights - weightsChange;

                    var biases = _layers[i].Biases;
                    var biasesChange = _learningRate / batchSize * nablaBiases[i];
                    _layers[i].Biases = biases - biasesChange;
                }
                #endregion

                if (isVerbose)
                {
                    Console.WriteLine($"Epoch - {epoch}," +
                                      $" error - {Math.Round(errorSum / batchSize, 2)}");
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