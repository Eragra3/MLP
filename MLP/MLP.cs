using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MLP.HelperFunctions;

namespace MLP
{
    public class MLP
    {
        private HiddenLayer[] layers;

        public MLP(params int[] sizes)
        {
            layers = new HiddenLayer[sizes.Length];

            for (int i = 0; i < sizes.Length - 1; i++)
            {
                layers[i] = new HiddenLayer(sizes[i], sizes[i + 1]);
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

            for (int i = 0; i < layers.Length; i++)
            {
                output = layers[i].Feedforward(output);
            }

            return Sigmoid(output);
        }

        public int GetLabel(float[] inputs)
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
    }
}
