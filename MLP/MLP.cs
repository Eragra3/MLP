using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
        public float[] Feedforward(float[] inputs)
        {
            float[] output = inputs;

            for (int i = 0; i < _layers.Length; i++)
            {
                output = _layers[i].Feedforward(output);
            }

            return Sigmoid(output);
        }

        public int Compute(float[] inputs)
        {
            var output = Feedforward(inputs);

            var max = output[0];
            var maxIndex = 0;

            for (int i = 1; i < output.Length; i++)
            {
                if (output[i] > max)
                {
                    max = output[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        public void Train(float[] inputs, int[] expectedOutput)
        {
            float[][][] nablaWeights = new float[_layers.Length][][];
            float[][] nablaBiases = new float[_layers.Length][];
            for (int i = 0; i < _layers.Length; i++)
            {
                var layerWeights = _layers[i]._weights;
                nablaWeights[i] = MatrixHelper.GetZerosMatrix(
                    layerWeights.Length,
                    layerWeights[0].Length
                    );
                nablaBiases[i] = MatrixHelper.GetZerosVector(layerWeights.Length);
            }

            float[][] outputs = new float[_layers.Length][];
            float[][] activations = new float[_layers.Length][];
            for (int i = 0; i < _layers.Length; i++)
            {
                outputs[i] = _layers[i].GetOutput(inputs);
                activations[i] = _layers[i].GetActivation(inputs);
            }

            var outputLayerIndex = _layers.Length - 1;
            var outputDelta = new float[outputs[outputs.Length - 1].Length];
            var outputLayerCostDerivative = SigmoidPrime(outputs[outputLayerIndex]);
            var outputLayerOutput = outputs[outputLayerIndex];
            for (int i = 0; i < outputDelta.Length; i++)
            {
                outputDelta[i] = (outputLayerOutput[i] - expectedOutput[i])
                    * outputLayerCostDerivative[i];
            }
            nablaBiases[outputLayerIndex] = outputDelta;
            nablaWeights[outputLayerIndex] = MatrixHelper.MultiplyVectors(
                outputDelta,
                activations[outputLayerIndex]
                );

            for (int layerIndex = _layers.Length - 2; layerIndex >= 0; layerIndex--)
            {
                var sigmoid_prime = Sigmoid(outputs[layerIndex]);
                var delta = 
            }
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