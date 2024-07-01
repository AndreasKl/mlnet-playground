using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
    public class HouseData
    {
        public float Size { get; set; }
        public float Year { get; set; }
        public float Price { get; set; }
    }

    public class Prediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }

    static void Main(string[] args)
    {
        HouseData[] houseData =
        [
            new()
            {
                Size = 1.1F,
                Year = 1980,
                Price = 1.0F
            },
            new()
            {
                Size = 1.9F,
                Year = 2018,
                Price = 3.3F
            },
            new()
            {
                Size = 2.8F,
                Year = 1960,
                Price = 1.6F
            },
            new()
            {
                Size = 3.4F,
                Year = 2022,
                Price = 4.2F
            },
            new()
            {
                Size = 3.4F,
                Year = 1928,
                Price = 1.2F
            },
            new()
            {
                Size = 2.4F,
                Year = 2024,
                Price = 4.4F
            },
        ];
        MLContext mlContext = new();
        IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

        var pipeline = mlContext
            .Transforms.Concatenate("Features", ["Size", "Year"])
            .Append(
                mlContext.Regression.Trainers.Sdca(
                    labelColumnName: "Price",
                    featureColumnName: "Features",
                    maximumNumberOfIterations: 150
                )
            );
        var model = pipeline.Fit(trainingData);

        // 4. Make a prediction
        var size = new HouseData() { Size = 2.5F, Year = 2024 };
        var price = mlContext
            .Model.CreatePredictionEngine<HouseData, Prediction>(model)
            .Predict(size);

        Console.WriteLine(
            $"Predicted price for size: {size.Size * 1000} sq ft= {price.Price * 100:C}k"
        );
    }
}
