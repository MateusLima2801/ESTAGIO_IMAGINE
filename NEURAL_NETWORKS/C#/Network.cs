using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork;

public class Network
{
    private int numLayers;
    private List<int> sizes;
    private List<double[,]> biases;
    private List<double[,]> weights;

    public Network(params int[] sizesParams)
    {
        var sizes = new List<int>(sizesParams);
        numLayers = sizes.Count;
        this.sizes = sizes;
        biases = new List<double[,]>();
        weights = new List<double[,]>();

        var random = new Random();

        for (int i = 1; i < numLayers; i++)
        {
            biases.Add(new double[sizes[i], 1]);
            for (int j = 0; j < sizes[i]; j++)
            {
                biases[i - 1][j, 0] = random.NextDouble();
            }

            weights.Add(new double[sizes[i], sizes[i - 1]]);
            for (int j = 0; j < sizes[i]; j++)
            {
                for (int k = 0; k < sizes[i - 1]; k++)
                {
                    weights[i - 1][j, k] = random.NextDouble();
                }
            }
        }
    }

    //     public Matrix FeedForward(Matrix a)
    //     {
    //         for (int i = 0; i < numLayers - 1; i++)
    //         {
    //             a = Sigmoid(weights[i] * a + biases[i]);
    //         }
    //         return a;
    //     }

    //     public void SGD(List<(Matrix, Matrix)> trainingData, int epochs, int miniBatchSize, double eta, List<(Matrix, Matrix)> testData = null)
    //     {
    //         int n = trainingData.Count;
    //         int nTest = testData?.Count ?? 0;

    //         for (int j = 0; j < epochs; j++)
    //         {
    //             trainingData = trainingData.OrderBy(x => Guid.NewGuid()).ToList(); // Shuffle the training data

    //             for (int k = 0; k < n; k += miniBatchSize)
    //             {
    //                 var miniBatch = trainingData.GetRange(k, Math.Min(miniBatchSize, n - k));
    //                 UpdateMiniBatch(miniBatch, eta);
    //             }

    //             if (testData != null)
    //             {
    //                 Console.WriteLine($"Epoch {j}: {Evaluate(testData)} / {nTest}");
    //             }
    //             else
    //             {
    //                 Console.WriteLine($"Epoch {j} complete");
    //             }
    //         }
    //     }

    //     private void UpdateMiniBatch(List<(Matrix, Matrix)> miniBatch, double eta)
    //     {
    //         List<Matrix> nablaB = new List<Matrix>();
    //         List<Matrix> nablaW = new List<Matrix>();

    //         for (int i = 0; i < numLayers - 1; i++)
    //         {
    //             nablaB.Add(new Matrix(sizes[i + 1], 1));
    //             nablaW.Add(new Matrix(sizes[i + 1], sizes[i]));
    //         }

    //         foreach (var (x, y) in miniBatch)
    //         {
    //             var (deltaNablaB, deltaNablaW) = Backprop(x, y);
    //             for (int i = 0; i < numLayers - 1; i++)
    //             {
    //                 nablaB[i] += deltaNablaB[i];
    //                 nablaW[i] += deltaNablaW[i];
    //             }
    //         }

    //         for (int i = 0; i < numLayers - 1; i++)
    //         {
    //             biases[i] -= (eta / miniBatch.Count) * nablaB[i];
    //             weights[i] -= (eta / miniBatch.Count) * nablaW[i];
    //         }
    //     }

    //     private (List<Matrix>, List<Matrix>) Backprop(Matrix x, Matrix y)
    //     {
    //         List<Matrix> nablaB = new List<Matrix>();
    //         List<Matrix> nablaW = new List<Matrix>();

    //         for (int i = 0; i < numLayers - 1; i++)
    //         {
    //             nablaB.Add(new Matrix(sizes[i + 1], 1));
    //             nablaW.Add(new Matrix(sizes[i + 1], sizes[i]));
    //         }

    //         Matrix activation = x;
    //         List<Matrix> activations = new List<Matrix> { x };
    //         List<Matrix> zs = new List<Matrix>();

    //         for (int i = 0; i < numLayers - 1; i++)
    //         {
    //             Matrix z = (weights[i] * activation) + biases[i];
    //             zs.Add(z);
    //             activation = Sigmoid(z);
    //             activations.Add(activation);
    //         }

    //         Matrix delta = CostDerivative(activations.Last(), y) * SigmoidPrime(zs.Last());
    //         nablaB[numLayers - 2] = delta;
    //         nablaW[numLayers - 2] = delta * activations[numLayers - 2].Transpose();

    //         for (int i = numLayers - 2; i > 0; i--)
    //         {
    //             Matrix z = zs[i - 1];
    //             Matrix sp = SigmoidPrime(z);
    //             delta = (weights[i].Transpose() * delta) * sp;
    //             nablaB[i - 1] = delta;
    //             nablaW[i - 1] = delta * activations[i - 1].Transpose();
    //         }

    //         return (nablaB, nablaW);
    //     }

    //     public int Evaluate(List<(Matrix, Matrix)> testData)
    //     {
    //         return testData.Count(t => FeedForward(t.Item1).Argmax() == t.Item2.Argmax());
    //     }

    //     private Matrix CostDerivative(Matrix outputActivations, Matrix y)
    //     {
    //         return outputActivations - y;
    //     }

    //     private Matrix Sigmoid(Matrix z)
    //     {
    //         return 1.0 / (1.0 + (-z).Exp());
    //     }

    //     private Matrix SigmoidPrime(Matrix z)
    //     {
    //         return Sigmoid(z) * (1.0 - Sigmoid(z));
    //     }

    //     public void PrintBiasesAndWeights()
    //     {
    //         Console.WriteLine("Biases:");
    //         for (int i = 0; i < biases.Count; i++)
    //         {
    //             Console.WriteLine($"Layer {i + 1} Biases:");
    //             Console.WriteLine(biases[i]);
    //         }

    //         Console.WriteLine("\nWeights:");
    //         for (int i = 0; i < weights.Count; i++)
    //         {
    //             Console.WriteLine($"Layer {i + 1} Weights:");
    //             Console.WriteLine(weights[i]);
    //         }
    //     }
    // }

    // public static class MatrixExtensions
    // {
    //     public static Matrix Exp(this Matrix matrix)
    //     {
    //         int rows = matrix.Rows;
    //         int cols = matrix.Columns;
    //         var result = new Matrix(rows, cols);

    //         for (int i = 0; i < rows; i++)
    //         {
    //             for (int j = 0; j < cols; j++)
    //             {
    //                 result[i, j] = Math.Exp(matrix[i, j]);
    //             }
    //         }

    //         return result;
    //     }

    //     public static int Argmax(this Matrix matrix)
    //     {
    //         int rows = matrix.Rows;
    //         int cols = matrix.Columns;
    //         if (rows != 1 && cols != 1)
    //         {
    //             throw new ArgumentException("Matrix should be 1D for Argmax operation.");
    //         }

    //         int maxIndex = 0;
    //         double maxValue = matrix[0, 0];

    //         for (int i = 0; i < rows; i++)
    //         {
    //             for (int j = 0; j < cols; j++)
    //             {
    //                 if (matrix[i, j] > maxValue)
    //                 {
    //                     maxIndex = i * cols + j;
    //                     maxValue = matrix[i, j];
    //                 }
    //             }
    //         }

    //         return maxIndex;
    //     }
}
