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

            var result = AlgebraHelper.Multiply(m1, m2);

            Assert.AreEqual(58, result[0][0]);
            Assert.AreEqual(64, result[0][1]);
            Assert.AreEqual(139, result[1][0]);
            Assert.AreEqual(154, result[1][1]);
        }
    }
}
