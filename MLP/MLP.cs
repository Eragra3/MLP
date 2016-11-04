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

        public void Train(float[] inputs, int expectedSolution)
        {
            var solution = Compute(inputs);



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
