using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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

        public float[] Feedforward(float[] inputs)
        {
            for (int i = 0; i < layers.Length; i++)
            {
                var layer = layers[i];

                layer.
            }
        }
    }
}
