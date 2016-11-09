using System;

namespace MLP.Extensions
{
    public static class RandomExtensions
    {
        public static float NextFloat(this Random rng)
        {
            return (float)(rng.NextDouble() * 2 - 1);
        }
    }
}
