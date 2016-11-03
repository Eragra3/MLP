using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public static class RandomExtensions
    {
        public static float NextFloat(this Random rng)
        {
            return (float)(rng.NextDouble() * 2 - 1);
        }
    }
}
