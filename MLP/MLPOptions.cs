﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public class MlpOptions
    {
        public double LearningRate { get; set; }

        public double Momentum { get; set; }

        public double ErrorThreshold { get; set; }

        public int MaxEpochs { get; set; }

        public int[] Sizes { get; set; }

        public string TrainingPath { get; set; }

        public string ValidationPath { get; set; }

        public string TestPath { get; set; }

        public MlpOptions(double learningRate, double momentum, double errorThreshold, int[] sizes, string trainingPath, string validationPath, string testPath, int maxEpochs)
        {
            this.LearningRate = learningRate;
            this.Momentum = momentum;
            this.ErrorThreshold = errorThreshold;
            this.Sizes = sizes;
            this.TrainingPath = trainingPath;
            this.ValidationPath = validationPath;
            this.TestPath = testPath;
            MaxEpochs = maxEpochs;
        }
    }
}