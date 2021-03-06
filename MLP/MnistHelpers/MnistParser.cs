﻿using System.Drawing;
using System.IO;
using MathNet.Numerics.LinearAlgebra.Double;
using MLP.Training;

namespace MLP.MnistHelpers
{
    public static class MnistParser
    {
        public static readonly char[] FileSeparators = { '/', '\\' };

        public static MnistImage ReadImage(string path, bool normalize)
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

                        int gray = (int)(color.R * 0.2126 + color.G * 0.7152 + color.B * 0.0722);

                        values[j + i * bitmap.Width] = 1 - gray / 255.0;
                    }
                }
            }

            var solution = new DenseVector(10);
            solution[label] = 1.0;

            if (normalize)
            {
                var max = values.Maximum();
                max /= 2;
                values.MapInplace(v => v - max);
            }

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

        public static MnistImage[] ReadAll(string pathToDirectory, bool normalize)
        {
            var directoryInfo = new DirectoryInfo(pathToDirectory);

            var files = directoryInfo.GetFiles("*.png");
            var count = files.Length;
            var models = new MnistImage[count];

            for (int i = 0; i < files.Length; i++)
            {
                models[i] = ReadImage(files[i].FullName, normalize);
            }

            return models;
        }
    }
}
