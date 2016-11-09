using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Newtonsoft.Json;
using static MLP.HelperFunctions;

namespace MLP
{
    public class MLP
    {
        [JsonProperty]
        private Layer[] _layers;

        [JsonProperty]
        private int[] _sizes;

        /// <summary>
        /// Should only be used to create deserialized object
        /// </summary>
        public MLP() { }

        public MLP(params int[] sizes)
        {
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

        public Tuple<Vector<double>[], Matrix<double>[]> Train(Vector<double> inputs, Vector<double> expectedOutput)
        {
            var layersCount = _layers.Length;
            var nablaWeights = new Matrix<double>[layersCount];
            var nablaBiases = new Vector<double>[layersCount];
            for (int i = 0; i < _layers.Length; i++)
            {
                var layerWeights = _layers[i]._weights;
                nablaBiases[i] = new DenseVector(layerWeights.RowCount);
            }

            var outputs = new Vector<double>[layersCount];
            var activations = new Vector<double>[layersCount];
            for (int i = 0; i < _layers.Length; i++)
            {
                outputs[i] = _layers[i].GetOutput(inputs);
                activations[i] = _layers[i].GetActivation(inputs);
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
                delta = _layers[layerIndex]._weights.Transpose().Multiply(delta).PointwiseMultiply(sigmoidPrime);
                nablaBiases[layerIndex] = delta;
                nablaWeights[layerIndex] = delta.OuterProduct(outputs[layerIndex + 1]);
            }

            var nablas = new Tuple<Vector<double>[], Matrix<double>[]>(nablaBiases, nablaWeights);

            return nablas;
        }

        public string ToJson()
        {
            return JsonConvert.SerializeObject(this, Formatting.Indented);
        }

        public static MLP FromJson(string json)
        {
            return JsonConvert.DeserializeObject<MLP>(json);
        }
    }
}