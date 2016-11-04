using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public static class MatrixHelper
    {
        private static Random _rng = new Random();

        /// <summary>
        /// Returned matrix is in column notation
        /// </summary>
        /// <param name="x">height of matrix</param>
        /// <param name="y">width of matrix</param>
        /// <returns></returns>
        public static float[][] GetRandomMatrix(int x, int y)
        {
            var matrix = new float[x][];
            for (int i = 0; i < matrix.Length; i++)
            {
                matrix[i] = new float[y];
                for (int j = 0; j < matrix[i].Length; j++)
                {
                    matrix[i][j] = _rng.NextFloat();
                }
            }
            return matrix;
        }

        public static float[] GetRandomVector(int x)
        {
            var vector = new float[x];
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = _rng.NextFloat();
            }
            return vector;
        }

        /// <summary>
        /// A x B  *  X x Y
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static float[][] Multiply(float[][] m1, float[][] m2)
        {
            var A = m1.Length;
            var B = m1[0].Length;
            var X = m2.Length;
            var Y = m2[0].Length;

            if (B != X)
            {
                var msg = $"Matrices have wrong dimensions! {A}x{B} {X}x{Y}";
                throw new ArgumentException(msg);
            }

            var result = new float[A][]; // A x Y
            for (int i = 0; i < A; i++)
            {
                result[i] = new float[Y];
                for (int j = 0; j < Y; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < B; k++)
                    {
                        sum += m1[i][k] * m2[k][j];
                    }
                    result[i][j] = sum;
                }
            }

            return result;
        }
    }
}
