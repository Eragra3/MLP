using System;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Storage;
using Newtonsoft.Json;

namespace MLP.Serialization
{
    public class DenseMatrixConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return typeof(DenseMatrix).IsAssignableFrom(objectType);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            var serializationModel = serializer.Deserialize<DenseMatrixSerializationModel>(reader);
            var storage = DenseColumnMajorMatrixStorage<double>.OfColumnMajorEnumerable(
                serializationModel.RowCount,
                serializationModel.ColumnCount,
                serializationModel.Storage
                );
            var matrix = new DenseMatrix(storage);
            return matrix;
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            var matrix = (DenseMatrix)value;
            var serializationModel = new DenseMatrixSerializationModel
            {
                RowCount = matrix.RowCount,
                ColumnCount = matrix.ColumnCount,
                Storage = matrix.Storage.ToColumnMajorArray()
            };
            serializer.Serialize(writer, serializationModel);
        }
    }
}
