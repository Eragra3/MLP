using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace MLP.MnistHelpers
{
    public static class MnistParser
    {
        public static readonly char[] FileSeparators = new[] { '/', '\\' };

        public static MnistImage ReadImage(string path)
        {
            Bitmap bitmap = new Bitmap(path);

            var fileName = path.Substring(path.LastIndexOfAny(FileSeparators) + 1);
            var label = int.Parse(fileName[0].ToString());

            var values = new DenseVector(bitmap.Height * bitmap.Width);

            for (int i = 0; i < bitmap.Height; i++)
            {
                for (int j = 0; j < bitmap.Width; j++)
                {
                    {
                        var color = bitmap.GetPixel(j, i);

                        var grayscale = (color.R + color.G + color.B) / 3;

                        values[j + i * bitmap.Width] = grayscale / 255.0;
                    }
                }
            }

            var solution = new SparseVector(10) { [label] = 1 };

            var image = new MnistImage
            {
                Width = bitmap.Width,
                Height = bitmap.Height,
                Values = values,
                FileName = fileName,
                Label = label,
                ExpectedSolution = solution
            };

            return image;
        }

        public static MnistImage[] ReadAll(string pathToDirectory)
        {
            var directoryInfo = new DirectoryInfo(pathToDirectory);

            var files = directoryInfo.GetFiles("*.png");
            var count = files.Length;
            var models = new MnistImage[count];

            for (int i = 0; i < files.Length; i++)
            {
                models[i] = ReadImage(files[i].FullName);
            }

            return models;
        }
    }
}
