using System;
using MathNet.Numerics.LinearAlgebra.Double;
using Newtonsoft.Json;

namespace MLP.Serialization
{
    public class DenseVectorConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return typeof(DenseVector).IsAssignableFrom(objectType);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            return new DenseVector(serializer.Deserialize<double[]>(reader));
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            serializer.Serialize(writer, value);
        }
    }
}
