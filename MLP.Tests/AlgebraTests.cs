using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MLP.Tests
{
    [TestClass]
    public class AlgebraTests
    {
        [TestMethod]
        public void Multiplicate()
        {
            var m1 = new[]
            {
                new float[]{1,2,3 },
                new float[]{4,5,6 }
            };

            var m2 = new[]
            {
                new float[]{7,8 },
                new float[]{9,10 },
                new float[]{11, 12 }
            };

            var result = MatrixHelper.Multiply(m1, m2);

            Assert.AreEqual(58, result[0][0]);
            Assert.AreEqual(64, result[0][1]);
            Assert.AreEqual(139, result[1][0]);
            Assert.AreEqual(154, result[1][1]);
        }

        [TestMethod]
        public void MultiplicateVectors_GetMatrix()
        {
            float[] m1 = { 1, 2, 3 };

            float[] m2T = { 4, 5, 6 };

            var result = MatrixHelper.MultiplyVectors(m1, m2T);

            Assert.AreEqual(4, result[0][0]);
            Assert.AreEqual(5, result[0][1]);
            Assert.AreEqual(6, result[0][2]);
            Assert.AreEqual(8, result[1][0]);
            Assert.AreEqual(10, result[1][1]);
            Assert.AreEqual(12, result[1][2]);
            Assert.AreEqual(12, result[2][0]);
            Assert.AreEqual(15, result[2][1]);
            Assert.AreEqual(18, result[2][2]);
        }
    }
}
