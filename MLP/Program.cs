using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLP.Cmd;
using TTRider.FluidCommandLine;

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
            int imageIndex = 1;

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
                    .DefaultParameter("path", path => outputPath = path, "Path to MNIST dat file")
                    .DefaultParameter("index", index => imageIndex = int.Parse(index), "Index of image you want to show")
                    .Option("v", () => isVerbose = true, "Explain what is happening")
                    .Option("verbose", () => isVerbose = true, "Explain what is happening")
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
                    Console.WriteLine($"Showing {neuralNetworkJson}");
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
