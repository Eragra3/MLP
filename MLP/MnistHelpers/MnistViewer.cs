﻿using System;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP.MnistHelpers
{
    public static class MnistViewer
    {
        public static string ToMatrix(int[] model, int width)
        {
            var max = model.Max();
            var maxDigits = (int)Math.Floor(Math.Log10(max) + 1);

            var sb = new StringBuilder();
            const string separator = "|";

            for (int i = 0; i < model.Length; i++)
            {
                sb.Append(model[i].ToString().PadRight(maxDigits));

                sb.Append((i + 1) % width == 0 ? "\n" : separator);
            }

            return sb.ToString();
        }
    }
}