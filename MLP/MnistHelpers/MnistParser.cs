﻿using System.Drawing;
using System.IO;

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

            var values = new int[bitmap.Height * bitmap.Width];

            for (int i = 0; i < bitmap.Height; i++)
            {
                for (int j = 0; j < bitmap.Width; j++)
                {
                    {
                        var color = bitmap.GetPixel(j, i);

                        var grayscale = (color.R + color.G + color.B) / 3;

                        values[j + i * bitmap.Width] = grayscale;
                    }
                }
            }

            var image = new MnistImage
            {
                Width = bitmap.Width,
                Height = bitmap.Height,
                Values = values,
                FileName = fileName,
                Label = label
            };

            return image;
        }

        public class MnistImage
        {
            public int[] Values;
            public int Height;
            public int Width;
            public int Label;
            public string FileName;
        }
    }
}
