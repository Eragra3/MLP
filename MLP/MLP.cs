﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MLP.Training;
using Newtonsoft.Json;

namespace MLP
{
    public class Mlp
    {
        [JsonProperty]
        private Layer[] _layers;

        [JsonProperty]
        private int[] _sizes;

        /// <summary>
        /// Should only be used to create deserialized object
        /// </summary>
        [JsonConstructor]
        private Mlp()
        {

        }

        public Mlp(
            ActivationFunction activationFunction,
            double normalStDev,
            params int[] sizes
            )
        {
            _sizes = sizes;
            _layers = new Layer[sizes.Length - 1];

            for (int i = 0; i < sizes.Length - 1; i++)
            {
                //var isLast = i == sizes.Length - 1;
                _layers[i] = new Layer(sizes[i], sizes[i + 1], activationFunction, normalStDev);
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
            var testSet = trainingModel.TestSet;
            var validationSet = trainingModel.ValidationSet;
            var errorTreshold = trainingModel.ErrorThreshold;
            var maxEpochs = trainingModel.MaxEpochs;
            var batchSize = trainingModel.BatchSize;
            var learningRate = trainingModel.LearningRate;
            var momentum = trainingModel.Momentum;

            var isVerbose = trainingModel.IsVerbose;
            var evaluateOnEachEpoch = trainingModel.EvaluateOnEachEpoch;

            IList<double> epochErrors = new List<double> { 0 };
            var epochEvaluations = new List<MlpTrainer.EvaluationModel>();

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
                var activationFunctions = _layers.Select(l => l.CurrentActivationFunction.ToString()).ToArray();
                var distributions = _layers.Select(l => l.CurrentDistribution.ToString()).ToArray();
                Console.WriteLine("Starting with params:");
                Console.WriteLine($"\tsizes- {JsonConvert.SerializeObject(_sizes)}");
                Console.WriteLine($"\tlearning rate - {learningRate}");
                Console.WriteLine($"\tmomentum- {momentum}");
                Console.WriteLine($"\terror threshold - {errorTreshold}");
                Console.WriteLine($"\tmax epochs - {maxEpochs}");
                Console.WriteLine($"\tactivation functions - {JsonConvert.SerializeObject(activationFunctions, Formatting.None)}");
                Console.WriteLine($"\tinitial weights distributions- {JsonConvert.SerializeObject(distributions, Formatting.None)}");
            }

            //Debugger.Launch();

            Matrix<double>[] prevWeightsChange = new Matrix<double>[layersCount];
            Vector<double>[] prevBiasChange = new Vector<double>[layersCount];

            MlpTrainer.EvaluationModel initialEvaluation = null;
            if (evaluateOnEachEpoch)
            {
                initialEvaluation = MlpTrainer.Evaluate(this, testSet);
                epochEvaluations.Add(initialEvaluation);
            }

            if (isVerbose)
            {
                var percentage = (initialEvaluation ?? MlpTrainer.Evaluate(this, testSet)).Percentage;
                Console.WriteLine($"Initial state, {percentage.ToString("#0.00")}");
            }

            #region log data
            //log data
            var path = "log.csv";

            StringBuilder log = new StringBuilder("sep=|");
            log.AppendLine();
            log.Append("epoch|evaluation_0|error_0");
            log.AppendLine();
            #endregion

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

                    errorSum += solution.Map2((y, o) => Math.Abs(y - o), expectedSolution).Sum();
                }
                errorSum /= batchSize;

                #region update parameters
                for (int i = 0; i < layersCount; i++)
                {
                    var weights = _layers[i].Weights;
                    var weightsChange = learningRate / batchSize * nablaWeights[i];
                    if (prevWeightsChange[i] != null) weightsChange += momentum * prevWeightsChange[i];
                    _layers[i].Weights = weights - weightsChange;

                    var biases = _layers[i].Biases;
                    var biasesChange = learningRate / batchSize * nablaBiases[i];
                    if (prevBiasChange[i] != null) biasesChange += momentum * prevBiasChange[i];
                    _layers[i].Biases = biases - biasesChange;

                    prevWeightsChange[i] = weightsChange;
                    prevBiasChange[i] = biasesChange;
                }
                #endregion

                MlpTrainer.EvaluationModel epochEvaluation = null;
                if (evaluateOnEachEpoch)
                {
                    epochEvaluation = MlpTrainer.Evaluate(this, testSet);
                    epochEvaluations.Add(epochEvaluation);
                }

                if (isVerbose)
                {
                    var percentage = (epochEvaluation ?? MlpTrainer.Evaluate(this, testSet)).Percentage;
                    Console.WriteLine($"Epoch - {epoch}," +
                                      $" error - {errorSum.ToString("#0.000")}," +
                                      $" test - {percentage.ToString("#0.00")}");
                }

                #region dump data
                var eval = (epochEvaluation ?? MlpTrainer.Evaluate(this, testSet)).Percentage;

                log.AppendLine(epoch + "|" + eval + "|" + errorSum);
                #endregion

                #region set nablas to zeroes
                for (int i = 0; i < layersCount; i++)
                {
                    nablaBiases[i].Clear();
                    nablaWeights[i].Clear();
                }
                #endregion

                epochErrors.Add(errorSum);
            }

            #region log data
            File.WriteAllText(path, log.ToString());
            #endregion

            var trainingResult = new TrainingResult
            {
                Mlp = this,
                Epochs = epoch,
                EpochErrors = epochErrors.ToArray(),
                Evaluations = epochEvaluations.ToArray()
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