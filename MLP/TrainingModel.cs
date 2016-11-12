﻿namespace MLP
{
    public class TrainingModel
    {
        public InputModel[] TrainingSet { get; set; }

        public InputModel[] ValidationSet { get; set; }

        public double ErrorThreshold { get; set; }

        public int MaxEpochs { get; set; }

        public bool IsVerbose { get; set; }

        public int BathSize { get; set; }
    }
}