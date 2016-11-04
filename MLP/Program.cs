using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLP.Cmd;
using MLP.MnistHelpers;
using TTRider.FluidCommandLine;
using System.IO;

namespace MLP
{
    class Program
    {
        static void Main(string[] args)
        {
            Command command = Command.Help;
            string neuralNetworkJson = "";
            bool isVerbose = false;
            string outputPath = "";
            string imagePath = "";
            bool print = false;

            ICommandLine commandLine = CommandLine
                .Help("h")
                .Help("help")
                .Command("test", () => command = Command.Test, true, "Test your MLP")
                    .DefaultParameter("input", json => neuralNetworkJson = json, "MLP data in json format", "Json")
                    .Option("v", () => isVerbose = true, "Explain what is happening")
                    .Option("verbose", () => isVerbose = true, "Explain what is happening")
                .Command("train", () => command = Command.Train, "Train new MLP")
                    .Parameter("output", path => outputPath = path, "Output file to save trained mlp")
                    .Option("v", () => isVerbose = true, "Explain what is happening")
                    .Option("verbose", () => isVerbose = true, "Explain what is happening")
                .Command("view", () => command = Command.View, "Show MNIST imag")
                    .DefaultParameter("path", path => imagePath = path, "Path to image")
                    .Option("p", () => print = true, "Display grayscale interpretation")
                    .Option("print", () => print = true, "Display grayscale interpretation")
                .End();

            commandLine.Run(args);

            switch (command)
            {
                case Command.Train:
                    Console.WriteLine($"Training and saving to {outputPath}");
                    break;
                case Command.Test:
                    Console.WriteLine($"Testing {neuralNetworkJson}");
                    break;
                case Command.View:
                    if (string.IsNullOrEmpty(imagePath))
                    {
                        Console.WriteLine($"Path to image not set");
                        return;
                    }
                    if (!File.Exists(imagePath))
                    {
                        Console.WriteLine($"File {imagePath} does not exist!");
                        return;
                    }

                    var model = MnistParser.ReadImage(imagePath);

                    if (print)
                    {
                        var modelMatrix = MnistViewer.ToMatrix(model.Values, model.Width);
                        Console.WriteLine(modelMatrix);
                    }
                    break;
                case Command.Help:
                    commandLine.Run("help");
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}
