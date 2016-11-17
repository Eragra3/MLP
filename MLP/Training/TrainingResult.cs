namespace MLP.Training
{
    public class TrainingResult
    {
        public Mlp Mlp { get; set; }

        public int Epochs { get; set; }

        public double[] EpochErrors { get; set; }
    }
}
