using System.Drawing;

namespace MLP.MnistHelpers
{
    public static class MnistParser
    {
        public static MnistImage ReadImage(string path)
        {
            Bitmap bitmap = new Bitmap(path);

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
                Values = values
            };

            return image;
        }

        public class MnistImage
        {
            public int[] Values;
            public int Height;
            public int Width;
        }
    }
}
